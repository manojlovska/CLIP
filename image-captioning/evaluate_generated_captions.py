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

def read_test_list(test_list_path):
    with open(test_list_path, 'r') as fr:
        # Read the lines of data as a single string
        test_images = fr.read()

        # Use method splitlines to split the lines 
        # of data at the newline delimiters
        test_images = test_images.splitlines()

    return test_images

gt_captions_path = "./data/captions/VGGFace2/captions_att_07052024.txt"
generated_captions_path = "./data/captions/VGGFace2/generated-captions/generated_captions_07052024_77.txt"
test_list_path = "../Datasets/VGGFace2/test_list.txt"
# generated_captions_path = "./data/captions/VGGFace2/captions_25_att_29032024.txt"

attributes_list = ['man', 'woman', 
                'young', 'middle_aged', 'senior', 
                'asian', 'white', 'black', 
                'bald', 'receding hairline', 'wavy', 'bangs', 
                'black hair', 'blond hair', 'brown hair',
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
test_images = read_test_list(test_list_path)

sum_acc = 0
count_gt_captions = 0
for key in tqdm(test_images):
    if key in gt_captions.keys():
        count_gt_captions += 1
        gt_caption = gt_captions[key]

        if key in generated_captions.keys():
            generated_caption = generated_captions[key]

            count = 0
            count_gt_att = 0
            for attribute in attributes_list:
                if attribute == "man":
                    start_idx_gt = gt_caption.find(attribute)
                    start_idx_gen = generated_caption.find(attribute)

                    if gt_caption[start_idx_gt-2:start_idx_gt] != "wo":
                        count_gt_att += 1

                        if generated_caption[start_idx_gen-2:start_idx_gen] != "wo":
                            count += 1
                    
                    # print(gt_caption)
                    # print(gt_caption[start_idx_gt-2:start_idx_gt])
                    # print(gt_caption[start_idx_gt-2:start_idx_gt] != "wo")
                    # print()
                    # print(generated_caption)
                    # print(generated_caption[start_idx_gen-2:start_idx_gen])
                    # print(generated_caption[start_idx_gen-2:start_idx_gen] != "wo")

                    # print(count_gt_att)
                    # print(count)

                    # break
                else:
                    if attribute in gt_caption:
                        count_gt_att += 1
                        if attribute in generated_caption:
                            count += 1
            
            caption_acc = count / count_gt_att
        else:
            print(f"Generated caption is empty for image {key} and capion accuracy is 0!")
            caption_acc = 0

        sum_acc += caption_acc
    else:
        continue

    # import pdb
    # pdb.set_trace()

mean_acc = sum_acc / count_gt_captions

save_path = "./data/captions/VGGFace2/generated-captions/evaluation-results/mean_acc_07052024_77.txt"
# Open the file in write mode and save the data
print(f"Writing the captions to: {save_path}")
with open(save_path, 'w') as file:
    file.write(f'mean_acc: {mean_acc}\n')