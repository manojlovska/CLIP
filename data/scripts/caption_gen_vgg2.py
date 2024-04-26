import re
import pandas as pd
from tqdm import tqdm

def convert_to_dictionary(data):
    data_dict = {}
    for line in data.split('\n'):
        if line.strip():
            key, value = line.split()
            data_dict[key] = int(value)
    return data_dict

def fix_caption(caption):
    # Remove comma if followed by space and full stop
    # Add a full stop after any comma followed by a space and a full stop
    # Remove duplicate full stops

    caption = re.sub(r',(?=\s*[A-Z])', '.', caption)
    caption = re.sub(r',\.', '.', caption)
    caption = re.sub(r'\.{2,}', '.', caption)
    return caption

def generate_caption(attributes):
    attribute_groups = {
        'Identity': ['Young', 'Middle_Aged', 'Senior'],
        'Ethnicity': ['Asian', 'White', 'Black'],
        'Appearance': ['Rosy_Cheeks', 'Shiny_Skin', 'Bald', 'Wavy_Hair', 'Receding_Hairline', 'Bangs', 
                       'Sideburns', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'No_Beard', 
                       'Mustache', '5_o_Clock_Shadow', 'Goatee', 'Oval_Face', 'Square_Face', 'Round_Face', 
                       'Double_Chin', 'High_Cheekbones', 'Chubby', 'Obstructed_Forehead', 'Fully_Visible_Forehead', 
                       'Brown_Eyes', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Arched_Eyebrows', 'Mouth_Closed', 
                       'Smiling', 'Big_Lips', 'Big_Nose', 'Pointy_Nose', 'Heavy_Makeup'],
        'Accessories': ['Wearing_Hat', 'Wearing_Earrings', 'Wearing_Necktie', 'Wearing_Lipstick', 'No_Eyewear', 'Eyeglasses'],
        'Attractiveness': ['Attractive']
    }
    caption_parts = {}
    for group, group_attributes in attribute_groups.items():
        group_caption_parts = []
        for attribute in group_attributes:
            if attributes[attribute] == 1:
                group_caption_parts.append(attribute.lower())
        if group_caption_parts:
            caption_parts[group] = group_caption_parts

    caption = "A photo of a"
    if attributes['Male'] == 1:
        gender = "man"
        pronoun = "He"
        poss_pronoun = "His"
    else:
        gender = "woman"
        pronoun = "She"
        poss_pronoun = "Her"

    # print(caption_parts)

    for i, (group, group_caption_parts) in enumerate(caption_parts.items()):
        if i == 0 and 'Identity' not in caption_parts:
            if 'Ethnicity' in caption_parts:
                caption += " " + gender + " of " + caption_parts["Ethnicity"][0] + " descent"
            else:
                caption += " " + gender + ""

        if group == 'Identity':
            if group_caption_parts:
                if 'Ethnicity' in caption_parts:
                    caption += " " + (', ').join(group_caption_parts) + ' ' + gender + " of " + caption_parts["Ethnicity"][0]  + " descent" # random.choice(group_caption_parts).lower()
                else:
                    caption += " " + (', ').join(group_caption_parts) + ' ' + gender + ""
            else:
                if 'Ethnicity' in caption_parts:
                    caption += " " + gender + " of " + caption_parts["Ethnicity"][0] + " descent"
                else:
                    caption += " " + gender + ""

        else:
            if group == 'Appearance':
                if "Bald".lower() in group_caption_parts:
                    caption += " that is bald."
                else:
                    if 'Wavy_Hair'.lower() in group_caption_parts:
                        caption += " with wavy locks"
                    if 'Bangs'.lower() in group_caption_parts:
                        caption += ", stylish bangs"
                    if 'Brown_Hair'.lower() in group_caption_parts:
                        caption += ", sporting brown hair"
                    if 'Black_Hair'.lower() in group_caption_parts:
                        caption += ", sporting black hair"
                    if 'Blond_Hair'.lower() in group_caption_parts:
                        caption += ", sporting blond hair"
                if 'Mustache'.lower() in group_caption_parts:
                    caption += ", with a distinguished mustache"
                if 'Oval_Face'.lower() in group_caption_parts:
                    caption += ". " + poss_pronoun + " face features an oval structure"
                if 'High_Cheekbones'.lower() in group_caption_parts:
                    caption += ", prominent cheekbones,"
                if 'Bushy_Eyebrows'.lower() in group_caption_parts:
                    caption += " and thick, graceful eyebrows."
                if 'Smiling'.lower() in group_caption_parts:
                    caption += " " + pronoun + " expresses joy with a radiant smile,"
                if 'Big_Nose'.lower() in group_caption_parts:
                    caption += " complemented by a prominent nose,"
                if 'Heavy_Makeup'.lower() in group_caption_parts:
                    caption += " and noticeable makeup."
                if 'No_Beard'.lower() in group_caption_parts:
                    caption += ". " + pronoun + " is clean-shaven,"
                if '5_o_Clock_Shadow'.lower() in group_caption_parts:
                    caption += " with a 5 o'clock shadow shave."
            elif group == 'Accessories':
                if 'Wearing_Earrings'.lower() in group_caption_parts:
                    caption += " " + pronoun + " is wearing elegant ear accessories"
                if 'Wearing_Necktie'.lower() in group_caption_parts:
                    caption += " and formal attire,"
            elif group == 'Attractiveness':
                caption += " giving off a charming vibe."
    return caption + "."

# Example usage:
path = '/mnt/hdd/volume1/MAAD-Face/MAAD_Face.csv'
labels_dataframe = pd.read_csv(path)
import pdb
pdb.set_trace()

print(f"Generating the captions ...")
captions = {}
for i in tqdm(range(len(labels_dataframe))):
    labels = labels_dataframe.loc[i].drop(["Filename"])
    filename = labels_dataframe.loc[i]["Filename"]

    data = labels.to_dict()

    caption = generate_caption(data)
    fixed_caption = fix_caption(caption)

    captions[filename] = fixed_caption

# Save the captions
output_file = '/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_25_att_29032024.txt'

# Open the file in write mode and save the data
print(f"Writing the captions to: {output_file}")
with open(output_file, 'w') as file:
    for image_name, caption in tqdm(captions.items()):
        file.write(f'{image_name} {caption}\n')


