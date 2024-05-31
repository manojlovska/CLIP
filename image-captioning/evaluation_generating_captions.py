import torch
from tqdm import tqdm
import clip
from experiments.vgg_face2_exp import VGGFace2Exp
import re
from safetensors.torch import load
from clip.model import build_model

def read_captions_as_dict(captions_path):
    image_dict = {}
    with open(captions_path, 'r') as file:
        for line in file:
            image_name, caption = line.strip().split(' ', 1)
            image_dict[image_name] = caption
    return image_dict

gt_captions_path = "./data/captions/VGGFace2/captions_att_07052024.txt"
# generated_captions_path = "./data/captions/VGGFace2/generated-captions/generated_captions_07052024_77.txt"
generated_captions_path = "./data/captions/VGGFace2/captions_25_att_29032024.txt"

attributes_list = ['man', 'woman', 
                'young', 'middle_aged', 'senior', 
                'asian', 'white', 'black', 
                'bald', 'receding hairline', 'wavy', 'bangs', 
                ' black hair', 'blond hair', 'brown hair',
                'arched', 'thick',
                'large',
                'heavy makeup',
                'smiles',
                'oval', 'chubby',
                'prominent cheekbones',
                'clean-shaven', 'mustache', 'shadow shave',
                'earrings', 'lipstick', 'hat',
                'visible forehead',
                'eyewear', 'eyeglasses',
                'formal attire']

gt_captions = read_captions_as_dict(gt_captions_path)
generated_captions = read_captions_as_dict(generated_captions_path)

mean_acc = 0
for key in gt_captions.keys():
    gt_caption = gt_captions[key]
    generated_caption = generated_captions[key]

    count = 0
    count_gt_att = 0
    for attribute in attributes_list:
        if attribute in gt_caption:
            count_gt_att += 1
            if attribute in generated_caption:
                count += 1
    
    caption_acc = count / count_gt_att

    mean_acc += caption_acc

    import pdb
    pdb.set_trace()

mean_acc = mean_acc / len(gt_captions)

import pdb
pdb.set_trace()



