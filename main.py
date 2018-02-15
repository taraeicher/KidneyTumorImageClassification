import glob
import os
import random as rand
from PIL import Image

from tqdm import tqdm

import config as cfg
from tiler import WholeSlideTiler


def main():
	# open input_path, and process each wholeslide image
	files = glob.glob(cfg.IMAGE_FOLDER_PATH + '/*.svs')
	for slidepath in tqdm(files):
		basename = os.path.splitext(os.path.basename(slidepath))[0]
		basepath = os.path.join(cfg.OUTPUT_FOLDER_PATH, basename)
		WholeSlideTiler(slidepath, basepath, cfg.IMG_FORMAT, cfg.TILE_SIZE, cfg.OVERLAP, cfg.LIMIT_BOUNDS, cfg.ROTATE,
						cfg.QUALITY, cfg.NUM_WORKERS, cfg.ONLY_LAST).run()
	displayTiles(cfg.OUTPUT_FOLDER_PATH, 10)

def displayTiles(path, display_count):
	#Display display_count images, selected randomly over all generated images.
	img_count = 0
	#Count all images in the slide directory that were not rejected.
	for root, dirs, files in os.walk(path):
		for file in files:   
			fullfile = os.path.join(root, file)
			if fullfile.endswith(cfg.IMG_FORMAT) and "slide" in fullfile and "rejected" not in fullfile:
				img_count += 1
				
	#Randomly choose indices to display.
	indices_to_display = []
	while len(indices_to_display) < display_count:
		idx = int(img_count * rand.random())
		if idx not in indices_to_display:
			indices_to_display.append(idx)
			
	#Loop through images and dispay the ones at the given indices.
	file_index = 0
	for root, dirs, files in os.walk(path):
		for file in files:    
			fullfile = os.path.join(root, file)
			if fullfile.endswith(cfg.IMG_FORMAT) and "slide" in fullfile and "rejected" not in fullfile:
				if file_index in indices_to_display:
					image = Image.open(fullfile)
					image.show()
				file_index += 1


if __name__ == '__main__':
	main()
