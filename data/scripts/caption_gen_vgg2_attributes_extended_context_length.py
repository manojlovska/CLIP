import re
import pandas as pd
from tqdm import tqdm
import clip

def convert_to_dictionary(data):
    data_dict = {}
    for line in data.split('\n'):
        if line.strip():
            key, value = line.split()
            data_dict[key] = int(value)
    return data_dict

# def fix_caption(caption):
#     # Remove comma if followed by space and full stop
#     # Add a full stop after any comma followed by a space and a full stop
#     # Remove duplicate full stops

#     caption = re.sub(r',(?=\s*[A-Z])', '.', caption)
#     caption = re.sub(r',\.', '.', caption)
#     caption = re.sub(r'\.{2,}', '.', caption)
#     return caption

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

def generate_caption(attributes):
    attribute_groups = {
        'Identity': ['Young', 'Middle_Aged', 'Senior'],
        'Ethnicity': ['Asian', 'White', 'Black'],
        'Appearance': ['Rosy_Cheeks', 'Shiny_Skin', 'Bald', 'Wavy_Hair', 'Receding_Hairline', 'Bangs', 
                       'Sideburns', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair', 'No_Beard', 
                       'Mustache', '5_o_Clock_Shadow', 'Goatee', 'Oval_Face', 'Square_Face', 'Round_Face', 
                       'Double_Chin', 'High_Cheekbones', 'Chubby', 'Obstructed_Forehead', 'Fully_Visible_Forehead', 
                       'Brown_Eyes', 'Bags_Under_Eyes', 'Bushy_Eyebrows', 'Arched_Eyebrows', 'Mouth_Closed', 
                       'Smiling', 'Big_Lips', 'Big_Nose', 'Pointy_Nose', 'Heavy_Makeup', 'Attractive'],
        'Accessories': ['Wearing_Hat', 'Wearing_Earrings', 'Wearing_Necktie', 'Wearing_Lipstick', 'No_Eyewear', 'Eyeglasses']
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

    for i, (group, group_caption_parts) in enumerate(caption_parts.items()):
        if i == 0 and 'Identity' not in caption_parts:
            if 'Ethnicity' in caption_parts:
                caption += " " + gender + " of " + caption_parts["Ethnicity"][0] + " descent"
            else:
                caption += " " + gender + ""

        if group == 'Identity':
            if group_caption_parts:
                if 'Ethnicity' in caption_parts:
                    caption += " " + (', ').join(group_caption_parts) + ' '+ caption_parts["Ethnicity"][0] + ' ' + gender
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
                    if 'Gray_Hair'.lower() in group_caption_parts:
                        caption += ", with gray hair"
                if 'Mustache'.lower() in group_caption_parts:
                    caption += ", with a distinguished mustache"
                if 'Goatee'.lower() in group_caption_parts:
                    caption += ", with a goatee"
                if 'Oval_Face'.lower() in group_caption_parts:
                    caption += ". " + pronoun + " has an oval face"
                if 'Square_Face'.lower() in group_caption_parts:
                    caption += ". " + pronoun + " has a square face"
                if 'Round_Face'.lower() in group_caption_parts:
                    caption += ". " + pronoun + " has a round face"
                if 'Double_Chin'.lower() in group_caption_parts:
                    caption += ". " + pronoun + " has a double chin"
                if 'High_Cheekbones'.lower() in group_caption_parts:
                    caption += ", prominent cheekbones,"
                if 'Chubby'.lower() in group_caption_parts:
                    caption += " with a chubby face."
                if 'Obstructed_Forehead'.lower() in group_caption_parts:
                    caption += " with a partially obstructed forehead."
                if 'Fully_Visible_Forehead'.lower() in group_caption_parts:
                    caption += " with a visible forehead."
                if 'Brown_Eyes'.lower() in group_caption_parts:
                    caption += " with brown eyes"
                if 'Bags_Under_Eyes'.lower() in group_caption_parts:
                    caption += ", with bags under the eyes"
                if 'Bushy_Eyebrows'.lower() in group_caption_parts:
                    caption += " and thick, graceful eyebrows."
                if 'Arched_Eyebrows'.lower() in group_caption_parts:
                    caption += " with arched eyebrows."
                if 'Mouth_Closed'.lower() in group_caption_parts:
                    caption += " with mouth closed."
                if 'Smiling'.lower() in group_caption_parts:
                    caption += " " + pronoun + " smiles joyfully,"
                if 'Big_Lips'.lower() in group_caption_parts:
                    caption += " with big lips,"
                if 'Big_Nose'.lower() in group_caption_parts:
                    caption += " with a large nose,"
                if 'Pointy_Nose'.lower() in group_caption_parts:
                    caption += " with a pointy nose,"
                if 'Heavy_Makeup'.lower() in group_caption_parts:
                    caption += " and heavy makeup."
                if 'No_Beard'.lower() in group_caption_parts:
                    caption += ". " + pronoun + " is clean-shaven,"
                if '5_o_Clock_Shadow'.lower() in group_caption_parts:
                    caption += "with a 5 o'clock shadow shave."
                if 'Receding_Hairline'.lower() in group_caption_parts:
                    caption += " with a receding hairline."
                if 'Mustache'.lower() in group_caption_parts:
                    caption += " with a mustache."
                if 'Attractive'.lower() in group_caption_parts:
                    caption += " with an attractive look."
            elif group == 'Accessories':
                if 'Wearing_Hat'.lower() in group_caption_parts:
                    caption += " wearing a stylish hat."
                if 'Wearing_Earrings'.lower() in group_caption_parts:
                    caption += " wearing elegant earrings"
                if 'Wearing_Necktie'.lower() in group_caption_parts:
                    caption += " and formal attire."
                if 'Wearing_Lipstick'.lower() in group_caption_parts:
                    caption += " with lipstick."
                if 'No_Eyewear'.lower() in group_caption_parts:
                    caption += " without any eyewear."
                if 'Eyeglasses'.lower() in group_caption_parts:
                    caption += " wearing eyeglasses."

    return caption + "."


# Example usage:
path = '/mnt/hdd/volume1/MAAD-Face/MAAD_Face.csv'
labels_dataframe = pd.read_csv(path)

print(f"Generating the captions ...")
captions = {}
for i in tqdm(range(len(labels_dataframe))): # len(labels_dataframe)
    filename = labels_dataframe.loc[i]["Filename"]
    labels = labels_dataframe.loc[i].drop(["Filename"])

    data = labels.to_dict()

    caption = generate_caption(data)
    fixed_caption = fix_caption(caption)

    tokenized_caption = clip.tokenize(fixed_caption, context_length=305, truncate=False)

    if tokenized_caption[0][-1].numpy().item() > 0:
        print("Context length exceeded!")
        import pdb
        pdb.set_trace()

    captions[filename] = fixed_caption
######################################################################################################################
######################################################################################################################
######################################################################################################################

import pdb
pdb.set_trace()

# Save the captions
output_file = '/home/anastasija/Documents/Projects/SBS/CLIP/data/captions/VGGFace2/captions_att_fixed_28052024.txt'

# Open the file in write mode and save the data
print(f"Writing the captions to: {output_file}")
with open(output_file, 'w') as file:
    for image_name, caption in tqdm(captions.items()):
        file.write(f'{image_name} {caption}\n')

import pdb
pdb.set_trace()

