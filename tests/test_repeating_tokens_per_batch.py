# from tqdm import tqdm
# from experiments.vgg_face2_exp import VGGFace2Exp
# from collections import Counter
# import json
# import multiprocessing as mp

# exp = VGGFace2Exp()

# print("Train Dataloader ...")
# train_loader = exp.gets_train_dataloader(batch_size=512)

# pbar = tqdm(train_loader, total=len(train_loader))

# relative_n_unique_captions_train = 0
# train_dict = {}
# print("Calculating the mean number of repeating values per batch (train set)...")
# for i, batch in enumerate(pbar):
#     image_names, images, captions, texts = batch

#     n_unique_captions_per_batch = len(tuple(set(captions)))
#     repeating_captions_per_batch = Counter(captions)

#     train_dict[i] = repeating_captions_per_batch

#     relative_n_unique_captions_train += n_unique_captions_per_batch

# relative_n_unique_captions_train /= len(train_loader)

# print("Writing to a json file ... ")
# with open("/ceph/grid/home/am6417/Thesis/CLIP/tests/results/train_dict.json", "w") as file:
#     json.dump(train_dict, file)

# print(f'relative_n_unique_captions_train: {relative_n_unique_captions_train}')

# #################################################################################################
# print("Val Dataloader ...")
# val_loader = exp.get_val_dataloader(batch_size=512)

# v_pbar = tqdm(val_loader, total=len(val_loader))

# relative_n_unique_captions_val = 0
# val_dict = {}
# print("Calculating the mean number of repeating values per batch (validation set)...")
# for i, batch in enumerate(v_pbar):
#     val_image_names, val_images, val_captions, val_texts = batch

#     n_unique_captions_per_batch = len(tuple(set(val_captions)))
#     repeating_captions_per_batch = Counter(val_captions)

#     val_dict[i] = repeating_captions_per_batch

#     relative_n_unique_captions_val += n_unique_captions_per_batch

# relative_n_unique_captions_val /= len(val_loader)

# print("Writing to a json file ... ")
# with open("/ceph/grid/home/am6417/Thesis/CLIP/tests/results/val_dict.json", "w") as file:
#     json.dump(val_dict, file)

# print(relative_n_unique_captions_val)

from tqdm import tqdm
from experiments.vgg_face2_exp import VGGFace2Exp
from collections import Counter
import json
from multiprocessing import Pool

def process_batch(batch):
    image_names, images, captions, texts = batch
    n_unique_captions_per_batch = len(set(captions))
    repeating_captions_per_batch = Counter(captions)
    return n_unique_captions_per_batch, repeating_captions_per_batch

def process_loader(loader):
    results = []
    with Pool() as pool:
        for result in tqdm(pool.imap_unordered(process_batch, loader), total=len(loader)):
            results.append(result)
    return results

exp = VGGFace2Exp()

print("Train Dataloader ...")
train_loader = exp.get_train_dataloader(batch_size=512)

print("Val Dataloader ...")
val_loader = exp.get_val_dataloader(batch_size=512)

train_results = process_loader(train_loader)
val_results = process_loader(val_loader)

train_dict = {i: result[1] for i, result in enumerate(train_results)}
val_dict = {i: result[1] for i, result in enumerate(val_results)}

relative_n_unique_captions_train = sum(result[0] for result in train_results) / len(train_loader)
relative_n_unique_captions_val = sum(result[0] for result in val_results) / len(val_loader)

print("Writing train results to a json file ... ")
with open("/ceph/grid/home/am6417/Thesis/CLIP/tests/results/train_dict.json", "w") as file:
    json.dump(train_dict, file)

print(f'relative_n_unique_captions_train: {relative_n_unique_captions_train}')

print("Writing val results to a json file ... ")
with open("/ceph/grid/home/am6417/Thesis/CLIP/tests/results/val_dict.json", "w") as file:
    json.dump(val_dict, file)

print(f'relative_n_unique_captions_val: {relative_n_unique_captions_val}')




