import heapq
import os
from operator import itemgetter
from unicodedata import normalize
from multiprocessing import Process, JoinableQueue

import sys

import shutil

import re

import time
import warnings

import PIL
import cv2
from PIL import Image
from openslide import open_slide, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import config as cfg
import numpy as np
warnings.simplefilter("error")


class TileWorker(Process):
	"""A child process that generates and writes tiles."""

	def __init__(self, queue, slidepath, tile_size, overlap, limit_bounds, rotate,
				 quality):
		Process.__init__(self, name='TileWorker')
		self.daemon = True
		self._queue = queue
		self._slidepath = slidepath
		self._tile_size = tile_size
		self._overlap = overlap
		self._limit_bounds = limit_bounds
		self._quality = quality
		self._rotate = rotate
		self._slide = None
		
	def normalize_staining(self, I):
		Io = 255 + 1
		beta = 0.01
		alpha = 1
		HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
		maxCRef = np.array([1.9705, 1.0308])
		
		(h, w, c) = np.shape(I)
		I = np.reshape(I, (h * w, c), order = 'F')
		# Step 1. Convert RGB to OD.\
		ones = np.ones(I.shape)
		OD = - np.log(np.add(I, ones) / Io) #optical density where each channel in the image is normalized to values between [0, 1]
		
		# Step 2. Remove data with OD intensity less than beta
		ODhat = (OD[(np.logical_not((OD < beta).any(axis = 1))), :])
		
		# Step 3. Calculate SVD on the OD tuples
		cov = np.cov(ODhat, rowvar = False)
		(W, V) = np.linalg.eig(cov)
		
		# Step 4. create plane from the SVD directions
		# corresponding to the two largest singular values
		Vec = - np.transpose(np.array([V[:, 1], V[:, 0]]))
		
		# Step 5. Project data onto the plane and normalize to unit Length
		That = np.dot(ODhat, Vec)
		
		# Step 6. Calculate angle of each point w.r.t the first SVD directions
		phi = np.arctan2(That[:, 1], That[:, 0])
		
		# Step 7. Find robust extremes (some alpha th and (100 - alpha th) percentiles of the angle
		minPhi = np.percentile(phi, alpha)
		maxPhi = np.percentile(phi, 100 - alpha)
		vMin = np.dot(Vec, np.array([np.cos(minPhi), np.sin(minPhi)]))
		vMax = np.dot(Vec, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
		if vMin[0] > vMax[0]:
			HE = np.array([vMin, vMax])
		else:
			HE = np.array([vMax, vMin])
		HE = np.transpose(HE)
		
		# Step 8. Convert extreme values back to OD space
		Y = np.transpose(np.reshape(OD, (h * w, c)))
		C = np.linalg.lstsq(HE, Y)
		maxC = np.percentile(C[0], 99, axis = 1)
		C = C[0] / maxC[:, None]
		C = C * maxCRef[:, None]
		hundreds = 100 * np.ones(np.dot(HERef, C).shape)
		point_ones = 0.01 * np.ones(np.dot(HERef, C).shape)
		Inorm = Io * np.exp(-1 * np.maximum(np.minimum(np.dot(HERef, C), hundreds), point_ones))
		Inorm = np.reshape(np.transpose(Inorm), (h, w, c), order = 'F')
		Inorm = np.clip(Inorm, 0, 255)
		Inorm = np.array(Inorm, dtype = np.uint8)
		
		return Inorm # ,H,E

	def run(self):
		self._slide = open_slide(self._slidepath)
		last_associated = None
		dz = self._get_dz()
		while True:
			data = self._queue.get()
			if data is None:
				self._queue.task_done()
				break

			associated, level, address, outfile, rejfile = data
			if last_associated != associated:
				dz = self._get_dz(associated)
				last_associated = associated

			tile = dz.get_tile(level, address)
			tile_norm = Image.fromarray(self.normalize_staining(np.asarray(tile)))

			if cfg.DONT_REJECT or self._is_good(tile_norm):
				tile_norm.save(outfile[:-5] + "_" + str(1) + outfile[-5:], quality=self._quality)

				if self._rotate:
					# 90 deg = 2, 180 deg = 3, 270 deg = 4
					for angle in [2, 3, 4]:
						self.rotate_and_save(tile_norm, angle, outfile)

			elif cfg.SAVE_REJECTED:
				tile_norm.save(rejfile, quality=self._quality)

			self._queue.task_done()
			
	def rotate_and_save(self, tile, angle_type, savefile):

		tile.transpose(angle_type).save(savefile[:-5] + "_" + str(angle_type) + savefile[-5:], quality=self._quality)

	def _get_dz(self, associated=None):
		if associated is not None:
			image = ImageSlide(self._slide.associated_images[associated])
		else:
			image = self._slide
		return DeepZoomGenerator(image, self._tile_size, self._overlap,
								 limit_bounds=self._limit_bounds)

	def _is_good(self, tile):
		# tile is PIL.image

		img = np.asarray(tile)

		if img.shape[0] < self._tile_size + 2 * self._overlap or img.shape[1] < self._tile_size + 2 * self._overlap:
			return False

		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(img, (5, 5), 0)
		ret3, th3 = cv2.threshold(blur, cfg.REJECT_THRESHOLD, 255, cv2.THRESH_BINARY)
		im2, contours, hierarchy = cv2.findContours(th3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		return self.get_cnt_sum(contours, 2) < cfg.MAX_WHITE_SIZE

	@staticmethod
	def get_cnt_sum(contours, topn):
		res = 0
		cnts = sorted(contours, key=lambda x: cv2.contourArea(x))[-topn:]
		return sum([cv2.contourArea(cnt) for cnt in cnts])


class SingleImageTiler(object):
	"""Handles generation of tiles and metadata for a single image."""

	def __init__(self, dz, basename, img_format, associated, queue, only_last=True):
		self._dz = dz
		self._basename = basename
		self._img_format = img_format
		self._associated = associated
		self._img_name = associated if associated else cfg.DEFAULT_FILENAME
		self._queue = queue
		self._processed = 0
		self._only_last = only_last

	def run(self):
		t = time.perf_counter()
		self._write_tiles()
		self._write_dzi()
		elapsed_time = time.perf_counter() - t
		cfg.ver_print("Tiling completed on {0} in: ".format(self._img_name), elapsed_time)

	def _write_tiles(self):
		if self._only_last:
			iterator = [self._dz.level_count - 1]
		else:
			iterator = range(self._dz.level_count)

		for level in iterator:

			tiledir = os.path.join(self._basename, self._img_name, str(level))
			rejpath = os.path.join(self._basename, self._img_name, str(level), "rejected")
			if not os.path.exists(tiledir):
				os.makedirs(tiledir)

			if not os.path.exists(rejpath) and cfg.SAVE_REJECTED:
				os.makedirs(rejpath)

			cols, rows = self._dz.level_tiles[level]

			for row in range(rows):
				for col in range(cols):
					tilename = os.path.join(tiledir, '%d_%d.%s' % (col, row, self._img_format))
					rejfile = os.path.join(rejpath, '%d_%d.%s' % (col, row, self._img_format))
					if not os.path.exists(tilename):
						self._queue.put((self._associated, level, (col, row), tilename, rejfile))

					self._tile_done()

	def _tile_done(self):
		self._processed += 1
		if self._only_last:
			ncols, nrows = self._dz.level_tiles[self._dz.level_count - 1]
			total = ncols * nrows
		else:
			total = self._dz.tile_count

		count = self._processed
		if count % 100 == 0 or count == total:
			print("\rTiling %s: wrote %d/%d tiles" % (
				self._associated or 'slide', count, total),
				  end='', file=sys.stderr)
			if count == total:
				print(file=sys.stderr)

	def _write_dzi(self):
		with open('%s.dzi' % self._basename, 'w') as fh:
			fh.write(self.get_dzi())

	def get_dzi(self):
		return self._dz.get_dzi(self._img_format)


class WholeSlideTiler(object):
	"""Handles generation of tiles and metadata for all images in a slide."""

	def __init__(self, slide_path, outpath, img_format, tile_size, overlap,
				 limit_bounds, rotate, quality, nworkers, only_last):

		self._slide = open_slide(slide_path)  # the whole slide image
		self._outpath = outpath  # baseline name of each tiled image
		self._img_format = img_format  # image format (jpeg or png)
		self._tile_size = tile_size  # tile size. default: 256x256 pixels
		self._overlap = overlap
		self._limit_bounds = limit_bounds
		self._queue = JoinableQueue(2 * nworkers)  # setup multiprocessing worker queues.
		self._nworkers = nworkers  # number of workers
		self._only_last = only_last
		self._dzi_data = {}
		for _i in range(nworkers):
			TileWorker(self._queue, slide_path, tile_size, overlap,
					   limit_bounds, rotate, quality).start()

	def run(self):
		self._run_image()
		for name in self._slide.associated_images:
			self._run_image(name)
			# self._write_static()
		self._shutdown()

	def _run_image(self, associated=None):
		"""Run a single image from self._slide."""
		if associated is None:
			image = self._slide
			outpath = self._outpath

		else:
			image = ImageSlide(self._slide.associated_images[associated])
			outpath = os.path.join(self._outpath, self._slugify(associated))

		dz = DeepZoomGenerator(image, self._tile_size, self._overlap, self._limit_bounds)

		tiler = SingleImageTiler(dz, outpath, self._img_format, associated,
								 self._queue, self._only_last)
		tiler.run()

		self._dzi_data[self._url_for(associated)] = tiler.get_dzi()

	def _url_for(self, associated):
		if associated is None:
			base = 'slide'
		else:
			base = self._slugify(associated)
		return '%s.dzi' % base

	@staticmethod
	def _copydir(src, dest):
		if not os.path.exists(dest):
			os.makedirs(dest)
		for name in os.listdir(src):
			srcpath = os.path.join(src, name)
			if os.path.isfile(srcpath):
				shutil.copy(srcpath, os.path.join(dest, name))

	@classmethod
	def _slugify(cls, text):
		text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
		return re.sub('[^a-z0-9]+', '_', text)

	def _shutdown(self):
		for _i in range(self._nworkers):
			self._queue.put(None)
		self._queue.join()
