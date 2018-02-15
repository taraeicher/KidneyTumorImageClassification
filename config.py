import os

# directory of the config file
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

DEFAULT_FILENAME = 'slide'

#IMAGE_FOLDER_PATH = "C:\\Users\\tarae\\OneDrive\\Documents\\PhD\\CSE_5599_imaging\\data_cleaning\\hist_imgs"
IMAGE_FOLDER_PATH = "C:\\Users\\tarae\\OneDrive\\Documents\\PhD\\CSE_5599_imaging\\single_hist_img"


#OUTPUT_FOLDER_PATH = "C:\\Users\\tarae\\OneDrive\\Documents\\PhD\\CSE_5599_imaging\\data_cleaning\\out"
OUTPUT_FOLDER_PATH = "C:\\Users\\tarae\\OneDrive\\Documents\\PhD\\CSE_5599_imaging\\out_large"

IMG_FORMAT = 'jpeg'

#TILE_SIZE = 149
TILE_SIZE = 1000

OVERLAP = 75

LIMIT_BOUNDS = True

QUALITY = 100

NUM_WORKERS = 12

ONLY_LAST = True

SAVE_REJECTED = True

DONT_REJECT = False

# increase this to reject more
REJECT_THRESHOLD = 200

ROTATE = True

MAX_WHITE_SIZE = (TILE_SIZE*TILE_SIZE)/2

def ver_print(string, value):
	print(string + " {0}".format(value))
