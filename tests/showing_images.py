import PIL
# from PIL import Image
import os
from IPython.display import Image

data_directory = "/mnt/hdd/volume1/anastasija/CLIP_outputs/CLIP-fine-tuning-improved-captions/robust-forest-20"
image = os.path.join(data_directory, "batch_val_img_1.png")
Image(filename=image) 

from PIL import Image
data_directory = "/mnt/hdd/volume1/anastasija/CelebA/CelebA/Img/img_celeba"
image = Image.open(os.path.join(data_directory, "162771.jpg"))
image.show()