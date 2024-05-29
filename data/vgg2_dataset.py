import os
from PIL import Image
import clip
from multiprocessing import Pool
from tqdm import tqdm
from loguru import logger

class VGGFace2Dataset():
    def __init__(self, vgg2_path, captions_path, preprocess,  split="train", num_workers=1):

        # Preprocessing of the images
        self.preprocess = preprocess

        # Base path
        self.base_path = vgg2_path

        # Images path
        self.images_path = os.path.join(self.base_path, "vggface2_mtcnn", "mtcnn_images")

        # Captions path
        self.captions_path = captions_path
        
        # Train or test split
        self.split = split
    
        # Images
        self.images_list = self.get_images_list()

        # Number of workers
        self.num_workers = num_workers

        # Initialize list to store indices of valid images
        self.valid_indices = self.get_valid_indices()

        # Captions for the particular split
        self.captions = self.read_captions_for_images()

        # logger.info(f"{len(self.valid_indices)}")

    def get_images_list(self):
        with open(os.path.join(self.base_path, f"{self.split}_list.txt"), 'r') as file:
            im_list = [line.strip() for line in file]
            
        return im_list
    
    def read_captions_for_images(self):
        matched_captions = {}
        # images_set = set(self.images_list)  # Convert images_list to a set for faster lookup
        images_set = set(self.images_list[i] for i in self.valid_indices)

        with open(self.captions_path, 'r') as file:
            for i, line in enumerate(file): # tqdm(file, desc="Processing captions")
                image_name, caption = line.strip().split(' ', 1)
                if image_name in images_set:
                    # logger.info(f"Image {image_name} is in images_set")
                    matched_captions[image_name] = caption
                # else:
                #     logger.info(f"Image {image_name} is not in images_set")

        return matched_captions
    
    # def get_valid_indices(self):
    #     # Populate valid_indices list with indices of non-None images
    #     valid_indices = []
    #     for idx in tqdm(range(len(self.images_list)), desc="Checking for valid indices"):
    #         im_path = os.path.join(self.images_path, self.images_list[idx])

    #         if os.path.exists(im_path):
    #             valid_indices.append(idx)

    #     return valid_indices

    def read_captions_as_dict(self):
        image_dict = {}
        with open(self.captions_path, 'r') as file:
            for line in file:
                image_name, caption = line.strip().split(' ', 1)
                image_dict[image_name] = caption
        return image_dict

    def get_valid_indices(self):
        valid_indices = []

        captions_dict = self.read_captions_as_dict()
        image_names = captions_dict.keys()

        for idx, image_name in enumerate(self.images_list): # tqdm(self.images_list, desc="Checking for valid indices")
            im_path = os.path.join(self.images_path, image_name)

            if os.path.exists(im_path) and image_name in image_names:
                valid_indices.append(idx)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        idx = self.valid_indices[index]

        # Preprocess image using CLIP's preprocessing function
        image_path = os.path.join(self.images_path, self.images_list[idx])

        image = self.load_image(image_path)
        caption = self.captions[self.images_list[idx]]

        tokenized_caption = clip.tokenize(caption, context_length=305, truncate=False)

        return self.images_list[idx], image, caption, tokenized_caption.squeeze()
    
    def load_image(self, image_path):
        # Define a function for loading image
        def load_image_worker(image_path):
            if not os.path.exists(image_path):
                return None
            else:
                return self.preprocess(Image.open(image_path))
            
        image = load_image_worker(image_path)

        return image