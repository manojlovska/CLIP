import torch
from tqdm import tqdm
import clip
from experiments.vgg_face2_exp import VGGFace2Exp
import re
from safetensors.torch import load
from clip.model import build_model
from torch.utils.data import DataLoader

DEVICE = "cpu"
BATCH_SIZE = 2  # Adjust based on your memory capacity

def load_model(state_dict, jit: bool = False):
    """ Modified clip.load function """
    if not jit:
        model = build_model(state_dict).to(DEVICE)
        if str(DEVICE) == "cpu":
            model.float()
        return model

    # patch the device names and dtype (same as your original code)

def swap_first_two_elements(lst):
    lst[0], lst[1] = lst[1], lst[0]
    return lst

def remove_periods_followed_by_comma(text):
    pattern = re.compile(r'\.\s*,')
    result = pattern.sub(',', text)
    return result

def fix_caption(caption):
    caption = re.sub(r',(?=\s*[A-Z])', '.', caption)
    caption = re.sub(r',\.', '.', caption)
    caption = re.sub(r'\.{2,}', '.', caption)
    caption = re.sub(r'\b(She|He),', r'\1', caption)
    pattern = re.compile(r'\.([a-z])|\. ([a-z])')
    
    def replace(match):
        if match.group(2):
            return ' ' + match.group(2)
        return ' ' + match.group(1)
    
    caption = pattern.sub(replace, caption)
    caption = remove_periods_followed_by_comma(caption)
    if not caption.endswith("."):
        caption = caption + "."
    return caption

def main():
    attribute_groups = {
        "sentence1": {
            'Gender': ['man', 'woman'],
            'Age': [' young', ' middle aged', ' senior'],
            'Ethnicity': [' of asian descent', ' of white descent', ' of black descent'],
            'Baldness': [' that is bald', ' with a receding hairline'],
            'HairType': [ 'with wavy locks', 'with stylish bangs'],
            'HairColor': [', sporting black hair.', ', sporting blond hair.', ', sporting brown hair.'],
            'Eyebrows': [' with arched eyebrows', ' with thick eyebrows', 'with thick arched eyebrows'],
            'Nose': [', with a large nose'],
            'Makeup': [' and a heavy makeup.']
        },
        "sentence2": {
            'Smiling': [' smiles joyfully'],
            'Face': [' has an oval face', ' has an oval chubby face'],
            'Cheekbones': [' with prominent cheekbones'],
        },
        "sentence3": {
            'Beard': [' is clean-shaven', ' has a mustache', ' has a 5 o clock shadow shave'],
            'Earrings': [' is wearing elegant earrings'],
            'Lipstick': [', is wearing a lipstick'],
            'Hat': [', is wearing a stylish hat'],
            'Forehead': [' , with a visible forehead'],
            'Eyeglasses': ['without any eyewear', ', wearing eyeglasses'],
            'Necktie': [' and a formal attire.']
        }
    }

    templates = [
        "A photo of a {}", 
        "A photo of a {} {}", 
        "A photo of a {} {} {}" 
    ]

    model_checkpoint = "CLIP_outputs/CLIP-fine-tuning-VGGFace2/fluent-moon-10/epoch_87/model.safetensors"
    with open(model_checkpoint, "rb") as f:
        data = f.read()
    loaded_state = load(data)
    model = load_model(loaded_state).to(DEVICE)

    exp = VGGFace2Exp()
    test_dataset = exp.get_val_dataset()
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    generated_dict = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image_names = batch[0]
            images = batch[1].to(DEVICE)
            # gt_captions = [item[2] for item in batch]

            batch_generated_captions = ["A photo of a "] * len(images)
            
            for sentence, attributes in attribute_groups.items():
                pronouns = [". She" if "woman" in caption else ". He" for caption in batch_generated_captions]
                if sentence != "sentence1":
                    batch_generated_captions = [cap + pron for cap, pron in zip(batch_generated_captions, pronouns)]

                for attribute, values in attributes.items():
                    if attribute == 'Gender':
                        previous_attributes = [""] * len(images)
                        attribute_captions = []
                        # for _ in range(len(images)):
                        attribute_captions.extend([templates[0].format(value) + "." for value in values])

                    elif attribute == 'Age':
                        attribute_captions = []
                        for i, cap in enumerate(batch_generated_captions):
                            attribute_captions.extend([templates[1].format(value, previous_attributes[i]) for value in values])

                    elif attribute == 'Ethnicity':
                        attribute_captions = []
                        for i, cap in enumerate(batch_generated_captions):
                            swapped_attributes = swap_first_two_elements([previous_attributes[i]] * 2)
                            attribute_captions.extend([templates[2].format(swapped_attributes[0], swapped_attributes[1], value) for value in values])

                    else:
                        attribute_captions = []
                        for cap in batch_generated_captions:
                            attribute_captions.extend([cap + value for value in values])

                    tokenized_captions = clip.tokenize(attribute_captions).to(DEVICE)
                    batch_logits = model(images, tokenized_captions) # images.unsqueeze(1).expand(-1, len(values), -1, -1, -1).reshape(-1, 3, images.shape[2], images.shape[3])
                    logits_per_image = batch_logits[0]
                    # import pdb
                    # pdb.set_trace()
                    probabilities = logits_per_image.softmax(dim=-1)

                    for i in range(len(images)):
                        max_prob, max_index = probabilities[i].max(dim=0)
                        if max_prob.cpu().numpy().item() - probabilities[i, 0].cpu().numpy().item() > 0.2:
                            if attribute in ['Gender', 'Age', 'Ethnicity']:
                                previous_attributes[i] = values[max_index.item()]
                            batch_generated_captions[i] = attribute_captions[max_index.item()]
                        else:
                            batch_generated_captions[i] = batch_generated_captions[i]

            for i, image_name in enumerate(image_names):
                generated_caption = fix_caption(batch_generated_captions[i])
                generated_dict[image_name] = generated_caption
        
            import pdb
            pdb.set_trace()

    save_path = "./data/captions/VGGFace2/generated-captions/generated_captions_07052024_77.txt"
    print(f"Writing the captions to: {save_path}")
    with open(save_path, 'w') as file:
        for image_name, caption in tqdm(generated_dict.items()):
            file.write(f'{image_name} {caption}\n')

if __name__ == "__main__":
    main()
