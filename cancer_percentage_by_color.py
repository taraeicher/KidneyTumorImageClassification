import os
import config as cfg
from PIL import Image
import numpy as np
from matplotlib import pyplot
import seaborn as sns

def main():
	#Loop through images and quantify the amount of blue.
	path = "C:\\Users\\tarae\\OneDrive\\Documents\\PhD\\CSE_5599_imaging\\data_cleaning\\out\\TCGA-61-1737-11A-01-TS1.75e75318-4d1f-4a7b-8258-6c32f463cb43\\slide\\14"
	plot_path = "C:\\Users\\tarae\\OneDrive\\Documents\\PhD\\CSE_5599_imaging\\color_ratios.png"
	ratio_white = []
	ratio_purple = []
	ratio_pink = []
	ratio_other = []
	tile_count = 0
	for root, dirs, files in os.walk(path):
		for file in files:    
			fullfile = os.path.join(root, file)
			if fullfile.endswith(cfg.IMG_FORMAT) and "slide" in fullfile and "rejected" not in fullfile:
				image = Image.open(fullfile)
				quantify_all(image, ratio_white, ratio_purple, ratio_pink, ratio_other)
				image.close()
				tile_count += 1
	plot_ratios(ratio_white, ratio_purple, ratio_pink, ratio_other, plot_path)
	quantify_total_cancer(ratio_purple, tile_count)

#Quantify and display amount of blue in each image
def quantify_all(img, ratio_white, ratio_purple, ratio_pink, ratio_other):
	rgb = img.convert('RGB')
	colors = np.asarray(rgb)
	blue_slices = colors[:,:,2]
	green_slices = colors[:,:,1]
	red_slices = colors[:,:,0]
	
	#Quantify amount of each relevant color.
	is_white = np.logical_and(blue_slices > 200, np.logical_and(green_slices > 200, red_slices > 200))
	is_pink = np.logical_and(red_slices > 50, np.logical_and(red_slices > blue_slices, np.logical_and(blue_slices > green_slices, np.logical_not(is_white))))
	is_purple = np.logical_and(blue_slices > 50, np.logical_and(blue_slices >= red_slices, np.logical_and(red_slices > green_slices, np.logical_not(is_white))))
	is_other = np.logical_and(np.logical_not(is_white), np.logical_and(np.logical_not(is_pink), np.logical_not(is_purple)))
	
	#Add color ratios to list.
	ratio_white.append(np.sum(is_white) / is_white.size)
	ratio_purple.append(np.sum(is_purple) / is_purple.size)
	ratio_pink.append(np.sum(is_pink) / is_pink.size)
	ratio_other.append(np.sum(is_other) / is_other.size)
	
#Plot ratios and save plots.
def plot_ratios(white, purple, pink, other, file):
	
	#Create density plot of all four ratios.
	all_color_plot = sns.kdeplot(white, color="k")
	all_color_plot = sns.kdeplot(pink, color="m")
	all_color_plot = sns.kdeplot(purple, color="b")
	all_color_plot = sns.kdeplot(other, color="g")
	fig = all_color_plot.get_figure()
	
	#Save the files.
	fig.savefig(file)
	
#Quantify the total amount of cancer by summing together the amount of cancer in all tiles.
def quantify_total_cancer(ratio_purple, tile_count):

	purple_sum = np.sum(ratio_purple)
	cancer_ratio = purple_sum / tile_count
	print(cancer_ratio)

if __name__ == '__main__':
	main()