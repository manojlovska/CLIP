import torch

import torch
import torch.nn.functional as F

def contrastive_loss(logits, pos_mask, temperature=0.5):
    # Normalize logits to get unit vectors
    logits_normalized = F.normalize(logits, p=2, dim=-1)
    
    # Compute pairwise cosine similarities
    similarities = torch.matmul(logits_normalized, logits_normalized.t()) / temperature
    
    # Extract the diagonal (positive pairs) and the complement of the mask (negative pairs)
    pos_pairs = similarities[pos_mask]
    neg_pairs = similarities[~pos_mask]
    
    # Compute contrastive loss
    numerator = torch.exp(pos_pairs)
    denominator = torch.exp(pos_pairs).sum() + torch.exp(neg_pairs).sum()
    loss = -torch.log(numerator / denominator)
    
    return loss.mean()

def get_num_diagonal_max_values(logits):
    probabilities = logits.softmax(dim=-1) # .cpu().numpy()

    max_values, max_values_indices = probabilities.max(dim=-1)
    diagonal_values = probabilities.diag()

    diagonal_max_values = max_values == diagonal_values

    return max_values_indices, diagonal_max_values, diagonal_max_values.sum()

# from experiments.base_exp import Exp

# exp = Exp()

# model = exp.get_model(exp.vision_encoder)

# optimizer = exp.get_optimizer()
# lr_scheduler = exp.get_lr_scheduler()

batch_size = 8


# Example usag
# logits_per_image = torch.randn(batch_size, 8)  # Example logits for 64 images
# logits_per_text = torch.randn(batch_size, 8)   # Example logits for 64 texts

logits_per_image = torch.tensor([[26.8594, 17.9062, 16.2969, 17.2969, 15.6094, 23.7188, 17.3906, 17.5938],
        [20.1719, 25.4375, 24.2500, 24.3594, 24.0625, 22.7344, 22.5469, 22.9531],
        [23.1562, 22.7812, 26.8281, 27.1719, 24.6094, 21.3125, 22.5469, 25.4688],
        [21.3125, 27.5000, 28.1562, 29.4688, 26.7188, 22.0625, 25.2031, 26.1250],
        [21.4062, 23.5938, 24.2344, 25.7500, 25.9062, 18.7344, 22.0625, 21.9688],
        [25.0938, 22.0469, 20.1562, 18.3281, 19.2031, 31.9531, 19.2344, 20.3594],
        [20.9844, 26.6719, 26.8750, 27.7969, 25.4844, 19.2812, 25.4688, 26.5938],
        [18.8125, 25.7344, 28.0781, 27.0469, 27.4688, 19.1094, 25.7500, 27.5469]])

logits_per_text = torch.tensor([[26.8594, 20.1719, 23.1562, 21.3125, 21.4062, 25.0938, 20.9844, 18.8125],
        [17.9062, 25.4375, 22.7812, 27.5000, 23.5938, 22.0469, 26.6719, 25.7344],
        [16.2969, 24.2500, 26.8281, 28.1562, 24.2344, 20.1562, 26.8750, 28.0781],
        [17.2969, 24.3594, 27.1719, 29.4688, 25.7500, 18.3281, 27.7969, 27.0469],
        [15.6094, 24.0625, 24.6094, 26.7188, 25.9062, 19.2031, 25.4844, 27.4688],
        [23.7188, 22.7344, 21.3125, 22.0625, 18.7344, 31.9531, 19.2812, 19.1094],
        [17.3906, 22.5469, 22.5469, 25.2031, 22.0625, 19.2344, 25.4688, 25.7500],
        [17.5938, 22.9531, 25.4688, 26.1250, 21.9688, 20.3594, 26.5938, 27.5469]])

max_values_indices_im, diagonal_max_values_im, num_diagonal_max_values_im = get_num_diagonal_max_values(logits_per_image)
max_values_indices_texts, diagonal_max_values_texts, num_diagonal_max_values_texts = get_num_diagonal_max_values(logits_per_text)

equal_diagonal_values = diagonal_max_values_im == diagonal_max_values_texts
num_equal_diagonal_values = equal_diagonal_values.sum()

equal_nondiagonal_values = max_values_indices_im[~equal_diagonal_values] == max_values_indices_texts[~equal_diagonal_values]
num_equal_nondiagonal_values = equal_nondiagonal_values.sum()

import pdb;
pdb.set_trace()
pos_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool)
# torch.diag(torch.tensor(1, (8, ), dtype=torch.bool), 0)

# Assuming a diagonal mask where each image pairs with its corresponding text
for i in range(batch_size):
    pos_mask[i, i] = True

# Compute the contrastive loss
loss_images = contrastive_loss(logits_per_image, pos_mask)
loss_texts = contrastive_loss(logits_per_text, pos_mask)


