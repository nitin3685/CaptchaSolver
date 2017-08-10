from scipy import misc
import glob

for image_path in glob.glob("/home/adam/*.png"):
    image = misc.imread(image_path)
    print image.shape
    print image.dtype
