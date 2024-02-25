import glob
from IPython.display import Image, display

HOME = '/Users/tanmay/Documents/Coding/Repositories/rubyrizz-data/'

for image_path in glob.glob(f'{HOME}runs/detect/predict2/*.jpg')[:3]:
      display(Image(filename=image_path, width=600))
      print("\n")