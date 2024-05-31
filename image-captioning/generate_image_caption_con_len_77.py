import torch
from tqdm import tqdm
import clip
from experiments.vgg_face2_exp import VGGFace2Exp
import re
from safetensors.torch import load
from clip.model import build_model

DEVICE = "cpu"


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

def swap_first_two_elements(lst):
    """
    Swap the first and second elements of the list.
    
    Args:
    lst (list): The list whose first and second elements are to be swapped.
    """
    lst[0], lst[1] = lst[1], lst[0]

    return lst

def remove_periods_followed_by_comma(text):
    # Use a regular expression to find periods followed by an optional space and then a comma
    pattern = re.compile(r'\.\s*,')
    
    # Replace the matched pattern with just a comma
    result = pattern.sub(',', text)
    
    return result

def fix_caption(caption):
    # Remove comma if followed by space and a capital letter
    caption = re.sub(r',(?=\s*[A-Z])', '.', caption)
    # Add a full stop after any comma followed by a space and a full stop
    caption = re.sub(r',\.', '.', caption)
    # Remove duplicate full stops
    caption = re.sub(r'\.{2,}', '.', caption)

    # Remove comma if it comes after "She" or "He"
    caption = re.sub(r'\b(She|He),', r'\1', caption)

    # Use a regular expression to find periods followed by a lowercase letter, with or without a space
    pattern = re.compile(r'\.([a-z])|\. ([a-z])')
    
    # Define a function to replace the matched pattern
    def replace(match):
        # If there's a space, replace with just the lowercase letter
        if match.group(2):
            return ' ' + match.group(2)
        # If there's no space, replace with just the lowercase letter
        return ' ' + match.group(1)
    
    # Substitute the matches with the replacement
    caption = pattern.sub(replace, caption)

    # Remove periods followed by a comma
    caption = remove_periods_followed_by_comma(caption)

    # Add full stop in the end if there is none
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
            'Face': [' has an oval face',
                    ' has an oval chubby face'],
            'Cheekbones': [' with prominent cheekbones'],
        },

        "sentence3": {
            'Beard': [' is clean-shaven', ' has a mustache', ' has a 5 o clock shadow shave'],
            'Earrings': [' is wearing elegant earrings'],
            'Lipstick': [', is wearing a lipstick'],
            'Hat': [', is wearing a stylish hat'],
            'Forehead': [' , with a visible forehead'],
            'Eyeglasses': ['without any eyewear', ', wearing eyeglasses'],
            'Necktie': [' and a formal attire.'] # [' and a formal attire.']
        }
    }

    templates = [
        "A photo of a {}", # A photo of a {man/woman}
        "A photo of a {} {}", # A photo of a {young/middle aged/senior} {man/woman}
        "A photo of a {} {} {}" # A photo of a {young/middle aged/senior} {man/woman} {of asian descent/of white descent/of black descent}
    ]

    # Build the model
    model_checkpoint = "CLIP_outputs/CLIP-fine-tuning-VGGFace2/fluent-moon-10/epoch_87/model.safetensors"
    with open(model_checkpoint, "rb") as f:
        data = f.read()
    loaded_state = load(data)
    model = load_model(loaded_state).to(DEVICE) # build_model(loaded_state).to(DEVICE)

    # Construct the dataset
    exp = VGGFace2Exp()
    test_set = exp.get_val_dataset()

    generated_dict = {}
    for idx in tqdm(range(test_set.__len__())):
        image_name = test_set[idx][0]
        image = test_set[idx][1].to(DEVICE)
        gt_caption = test_set[idx][2]

        # Generate the caption
        start_caption = "A photo of a "
        previous_caption = start_caption

        generated_caption = ""
        for sentence in attribute_groups.keys():

            if sentence != "sentence1":
                if "woman" in previous_caption:
                    pronoun = ". She"
                else:
                    pronoun = ". He"
                
                previous_caption += pronoun

                # import pdb
                # pdb.set_trace()

            previous_attributes = []
            for attribute in attribute_groups[sentence].keys():
                if attribute == "Gender":
                    previous_caption = templates[0]

                    attribute_captions = []
                    for value in attribute_groups[sentence][attribute]:
                        attribute_captions.append(previous_caption.format(value) + ".")
                    
                    tokenized_captions = clip.tokenize(attribute_captions).to(DEVICE)

                    # import pdb
                    # pdb.set_trace()

                    with torch.no_grad():
                        model.eval()
                        logits_per_image, _ = model(image.unsqueeze(dim=0), tokenized_captions)
                        probabilities = logits_per_image.softmax(dim=-1)
                        max_prob, max_index = probabilities.max(dim=-1)

                    previous_attributes.append(attribute_groups[sentence][attribute][max_index.cpu().numpy().item()])
                    previous_caption = attribute_captions[max_index.cpu().numpy().item()]

                elif attribute == "Age":
                    previous_caption = templates[1]

                    attribute_captions = []
                    for value in attribute_groups[sentence][attribute]:
                        attribute_captions.append(previous_caption.format(value, previous_attributes[0]))
                    
                    tokenized_captions = clip.tokenize(attribute_captions).to(DEVICE)

                    # import pdb
                    # pdb.set_trace()

                    with torch.no_grad():
                        model.eval()
                        logits_per_image, _ = model(image.unsqueeze(dim=0), tokenized_captions)
                        probabilities = logits_per_image.softmax(dim=-1)
                        max_prob, max_index = probabilities.max(dim=-1)

                    previous_attributes.append(attribute_groups[sentence][attribute][max_index.cpu().numpy().item()])
                    previous_caption = attribute_captions[max_index.cpu().numpy().item()]


                elif attribute == "Ethnicity":
                    previous_caption = templates[2]

                    attribute_captions = []
                    for value in attribute_groups[sentence][attribute]:
                        attribute_captions.append(previous_caption.format(*swap_first_two_elements(previous_attributes), value))
                    
                    tokenized_captions = clip.tokenize(attribute_captions).to(DEVICE)

                    # import pdb
                    # pdb.set_trace()

                    with torch.no_grad():
                        model.eval()
                        logits_per_image, _ = model(image.unsqueeze(dim=0), tokenized_captions)
                        probabilities = logits_per_image.softmax(dim=-1)
                        max_prob, max_index = probabilities.max(dim=-1)

                    previous_attributes.append(attribute_groups[sentence][attribute][max_index.cpu().numpy().item()])
                    previous_caption = attribute_captions[max_index.cpu().numpy().item()]

                else:
                    attribute_captions = [previous_caption] if previous_caption != start_caption else []
                    for value in attribute_groups[sentence][attribute]:
                        attribute_captions.append(previous_caption + value)
                    
                    tokenized_captions = clip.tokenize(attribute_captions).to(DEVICE)

                    # import pdb
                    # pdb.set_trace()

                    with torch.no_grad():
                        model.eval()
                        logits_per_image, _ = model(image.unsqueeze(dim=0), tokenized_captions)
                        probabilities = logits_per_image.softmax(dim=-1)
                        max_prob, max_index = probabilities.max(dim=-1)

                    # import pdb
                    # pdb.set_trace()

                    # If the max probability is more than 0.2 higher that the probability of the previous caption,
                    # add the feature
                    # If not the model is not that sure about adding a new feature, so we keep the previous caption
                    if max_prob.cpu().numpy()[0] - probabilities.cpu().numpy()[0][0] > 0.2:
                        selected_caption = attribute_captions[max_index.cpu().numpy().item()]
                    else:
                        selected_caption = previous_caption

                    generated_caption = selected_caption  # Update only the new part
                    generated_caption = fix_caption(generated_caption) # Fix the caption
                    previous_caption = generated_caption  # Update for next iteration

        generated_caption = fix_caption(generated_caption) # Fix the caption

        generated_dict[image_name] = generated_caption


    save_path = "./data/captions/VGGFace2/generated-captions/generated_captions_07052024_77.txt"
    # Open the file in write mode and save the data
    print(f"Writing the captions to: {save_path}")
    with open(save_path, 'w') as file:
        for image_name, caption in tqdm(generated_dict.items()):
            file.write(f'{image_name} {caption}\n')
    
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()