import os
import torch
from loguru import logger
from tqdm import tqdm
import clip
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from torch.utils.data import DataLoader
from safetensors.torch import load as load_safetensors
# from clip.model import build_model
from clip.model_extended import build_model
from torch import nn

# Set the device
DEVICE = "cuda:1"

def load_model(state_dict, jit: bool = False):
    """ Modified clip.load function """
    if not jit:
        model = build_model(state_dict).to(DEVICE)
        if str(DEVICE) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(DEVICE)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def _node_get(node: torch._C.Node, key: str):
        """Gets attributes of a node which is polymorphic over return type.
        
        From https://github.com/pytorch/pytorch/pull/82628
        """
        sel = node.kindOf(key)
        return getattr(node, sel)(key)

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(_node_get(node, "value")).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(DEVICE) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if _node_get(inputs[i].node(), "value") == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model

def map_split2int(split):
    """ Map the split of the data to an integer for reaading from annotatons file 
        - train: 0
        - val:   1
        - test:  2 
    """
    map_dict = {"train": 0,
                "val": 1,
                "test": 2}
    return map_dict[split]


def read_eval_partition(eval_partition_path, split_name="val"):
    split_int = map_split2int(split_name)
    with open(os.path.join(eval_partition_path, "list_eval_partition.txt"), 'r') as file:
        # Initialize an empty list to store image names with 0
        image_list = []

        # Iterate through each line in the file
        for line in file:
            # Split the line into image filename and integer
            image_filename, split = line.split()

            # Check if the integer is 0
            if int(split) == split_int:
                # Add the image filename to the list
                image_list.append(image_filename)

    return image_list


def generate_zero_shot_scores(captions_path, images_path, eval_partition_path, save_filename):
    # Load validation images
    image_list = read_eval_partition(eval_partition_path)
    val_images = [os.path.join(images_path, image) for image in image_list]

    # Load the model
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    attr_captions = []

    with open(captions_path, 'r') as file:
        for line in file:
            attr_captions.append(line.strip())

    with torch.no_grad():
        result = {}
        pbar = tqdm(val_images, total=len(val_images))
        model.eval()
        result = np.empty(shape=(len(val_images),40))
        idx = 0
        for img in pbar:
            img_name = os.path.basename(img)
            image = preprocess(Image.open(img))
            image = image.unsqueeze(0).to(DEVICE)
            cosine_similarities = []
            for attr_caption in attr_captions:
                # attr_caption = attr_captions[0] ############################## Comment this
                tokenized_caption = clip.tokenize(attr_caption, truncate=True)

                text = tokenized_caption.to(DEVICE)

                logits_per_image, _ = model(image, text)
                cosine_sim = logits_per_image.item() / 100.
                cosine_similarities.append(cosine_sim)
            result[idx] = cosine_similarities
            idx += 1

    np.save(save_filename, result)

def generate_zero_shot_scores_VGGFace2(captions_path, save_filename):
    # Load validation images
    # from experiments.vgg_face2_exp import VGGFace2Exp
    from data.vgg2_dataset import VGGFace2Dataset
    from clip.clip import load

    image_captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_att_07052024.txt"
    vgg2_path = "/mnt/hdd/volume1/VGGFace2"

    # Load the model
    model, preprocess = load("ViT-B/32",device=DEVICE,jit=False)
    model = model.to(DEVICE)

    # Val set
    val_set = VGGFace2Dataset(vgg2_path=vgg2_path, captions_path=image_captions_path, preprocess=preprocess, split="test")
    val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False)

    attr_captions = []

    with open(captions_path, 'r') as file:
        for line in file:
            attr_captions.append(line.strip())
    
    tokenized_captions = clip.tokenize(attr_captions)
    texts = tokenized_captions.to(DEVICE)

    with torch.no_grad():
        # result = {}
        pbar = tqdm(val_dataloader, total=len(val_dataloader))
        model.eval()
        result = np.empty(shape=(len(val_set),47))
        idx = 0
        for batch in pbar:
            image_names, images, _, _ = batch
            images = images.to(DEVICE)

            logits_per_image, _ = model(images, texts)
            cosine_similarities = logits_per_image / 100.

            result[idx*64:(idx+1)*64] = cosine_similarities.cpu().numpy()
            idx += 1

    import pdb
    pdb.set_trace()
    np.save(save_filename, result)

    import pdb
    pdb.set_trace()

def generate_fine_tuned_scores_VGGFace2(captions_path, image_captions_path, model_checkpoint, save_filename):
    # Load validation images
    # from experiments.vgg_face2_exp import VGGFace2Exp
    from data.vgg2_dataset import VGGFace2Dataset
    from clip.clip import load

    # image_captions_path = "/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_att_07052024.txt"
    vgg2_path = "/mnt/hdd/volume1/VGGFace2"

    # Load the model
    _, preprocess = load("ViT-B/32",device=DEVICE,jit=False)

    # Build the model
    # model_checkpoint = "CLIP_outputs/CLIP-fine-tuning-VGGFace2/fluent-moon-10/epoch_87/model.safetensors"
    with open(model_checkpoint, "rb") as f:
        data = f.read()
    loaded_state = load_safetensors(data)
    model = load_model(loaded_state).to(DEVICE) # build_model(loaded_state).to(DEVICE)

    # Val set
    val_set = VGGFace2Dataset(vgg2_path=vgg2_path, captions_path=image_captions_path, preprocess=preprocess, split="test")
    val_dataloader = DataLoader(val_set, batch_size=64, shuffle=False)

    attr_captions = []

    with open(captions_path, 'r') as file:
        for line in file:
            attr_captions.append(line.strip())
    
    tokenized_captions = clip.tokenize(attr_captions, context_length=305)
    texts = tokenized_captions.to(DEVICE)

    with torch.no_grad():
        # result = {}
        pbar = tqdm(val_dataloader, total=len(val_dataloader))
        model.eval()
        result = np.empty(shape=(len(val_set),47))
        idx = 0
        for batch in pbar:
            image_names, images, _, _ = batch
            images = images.to(DEVICE)

            logits_per_image, _ = model(images, texts)
            cosine_similarities = logits_per_image / 100.

            result[idx*64:(idx+1)*64] = cosine_similarities.cpu().numpy()
            idx += 1

    import pdb
    pdb.set_trace()
    
    np.save(save_filename, result)

    import pdb
    pdb.set_trace()

def generate_zero_shot_scores2(captions_path, images_path, eval_partition_path, save_filename):
    # Load validation images
    image_list = read_eval_partition(eval_partition_path)
    val_images = [os.path.join(images_path, image) for image in image_list]

    # Load the model
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    attr_captions1 = []

    with open(os.path.join(captions_path,"captions_combined_attr_man.txt"), 'r') as file:
        for line in file:
            attr_captions1.append(line.strip())
    
    attr_captions2 = []

    with open(os.path.join(captions_path,"captions_combined_attr_woman.txt"), 'r') as file:
        for line in file:
            attr_captions2.append(line.strip())

    with torch.no_grad():
        result = {}
        pbar = tqdm(val_images, total=len(val_images))
        model.eval()
        result = np.empty(shape=(len(val_images), len(attr_captions1), 2))
        idx = 0
        for img in pbar:
            img_name = os.path.basename(img)
            image = preprocess(Image.open(img))
            image = image.unsqueeze(0).to(DEVICE)
            cosine_similarities = []
            for i in range(len(attr_captions1)):
                # attr_caption = attr_captions[0] ############################## Comment this
                attr_caption1 = attr_captions1[i]
                attr_caption2 = attr_captions2[i]
                tokenized_caption = clip.tokenize([attr_caption1, attr_caption2], truncate=True)

                text = tokenized_caption.to(DEVICE)

                logits_per_image, _ = model(image, text)
                cosine_sim = logits_per_image.cpu().numpy() / 100.
                cosine_similarities.append(cosine_sim.flatten())
                # logger.info(f'cosine sim: {cosine_sim}')
                # logger.info(f'cosine similarities: {cosine_similarities}')

            result[idx] = cosine_similarities
            idx += 1

    np.save(save_filename, result)

def generate_zero_shot_scores_combined_attr(captions_path, images_path, eval_partition_path, save_filename):
    # Load validation images
    image_list = read_eval_partition(eval_partition_path)
    val_images = [os.path.join(images_path, image) for image in image_list]

    # Load the model
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    attr_captions = []

    with open(captions_path,'r') as file:
        for line in file:
            attr_captions.append(line.strip())

    tokenized_captions = clip.tokenize(attr_captions, truncate=True)
    text = tokenized_captions.to(DEVICE)

    save_file = os.path.join(save_filename, captions_path.split('/')[-2], captions_path.split('/')[-1].split('.')[-2])
    logger.info(f'save_path: {save_file}')
    dir = os.path.join(*save_file.split('/')[:-1])
    logger.info(f'makedirs: {dir}')
    
    logger.info(f'save_path: {save_file}')

    with torch.no_grad():
        result = {}
        pbar = tqdm(val_images, total=len(val_images))
        model.eval()
        result = []# np.empty(shape=(len(val_images), len(attr_captions)))
        idx = 0
        for img in pbar:
            img_name = os.path.basename(img)
            image = preprocess(Image.open(img))
            image = image.unsqueeze(0).to(DEVICE)
            cosine_similarities = []

            logits_per_image, _ = model(image, text)
            cosine_sim = logits_per_image.cpu().numpy() / 100.
            cosine_similarities.append(cosine_sim.flatten())
            # logger.info(f'cosine sim: {cosine_sim}')
            # logger.info(f'cosine similarities: {cosine_similarities}')

            result.append(cosine_similarities)
            # logger.info(result)
            idx += 1

    # os.makedirs(os.path.join(*save_file.split('/')[:-1]), exist_ok=True)
    np.save(save_file, result)


def get_annotation(fnmtxt, columns=None, verbose=True):
    if verbose:
        print("_"*70)
        print(fnmtxt)
    
    rfile = open(fnmtxt, 'r' ) 
    texts = rfile.readlines()
    rfile.close()
    
    if not columns:
        columns = np.array(texts[1].split(" "))
        columns = columns[columns != "\n"]
        texts = texts[2:]
    
    df = []
    for txt in texts:
        txt = np.array(txt.rstrip("\n").split(" "))
        txt = txt[txt != ""]
    
        df.append(txt)
        
    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"]+ list(columns)
    df.columns = columns   
    df = df.dropna()
    if verbose:
        print(" Total number of annotations {}\n".format(df.shape))
        print(df.head())
    ## cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm],downcast="integer")
    return(df)

# Get the ground truths
def get_gt(annotations_path, start_idx=162770, end_idx=182637):
    attr = get_annotation(os.path.join(annotations_path, 'list_attr_celeba.txt'), verbose=False)
    val_attr = attr.iloc[start_idx:end_idx]

    gt_all = np.empty(shape=(len(val_attr),40))
    for i in range(len(val_attr)):
        gt_all[i] = list(val_attr.iloc[i][1:])

    gt_all = np.where(gt_all == -1, 0, gt_all)

    return gt_all

# Calculate ROC curves
def roc(predicted, gt):
    roc_curves = []
    auc_scores = []
    for i in range(predicted.shape[1]):
        fpr, tpr, _ = roc_curve(gt[:, i], predicted[:, i])
        roc_auc = auc(fpr, tpr)
        roc_curves.append((fpr, tpr))
        auc_scores.append(roc_auc)
    
    return roc_curves, auc_scores

