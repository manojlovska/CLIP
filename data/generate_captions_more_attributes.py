import os
import glob
import shutil

import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import clip
"""
Example of the longest caption:
    A photo of an attractive, young woman with straight black hair. 
    She has a pale skin and an oval face with arched eyebrows, pointy nose, high cheekbones and narrow eyes with bags under the eyes. 
    She is smiling with big lips and she is wearing heavy makeup, a necklace and a necktie. 
    She has eyeglasses and a beard.
"""

first_sentence_templates = [
"A photo of a {}.",                     # -> A photo of a {man/woman} if [attractive, young, straight/wavy/bald hair, black/blond/brown/gray hair, pale skin] are not present
"A photo of {} {} {}.",                 # -> A photo of an attractive young man/woman.
"A photo of {} {}.",                    # -> A photo of an attractive/a young man/woman.

"A photo of a {} with {} {} hair.",     # -> A photo of a man/woman with straight/wavy black/blond/brown/gray hair.
"A photo of a {} with {} hair.",        # -> A photo of a man/woman with straight/wavy/black/blond/brown/gray hair.
"A photo of a {} that is {}.",          # -> A photo of a man/woman that is bald
"A photo of {} {} {} with {} {} hair.", # -> A photo of an attractive young man/woman with straight/wavy black/blond/brown/gray hair.
"A photo of {} {} {} with {} hair.",    # -> A photo of an attractive young man/woman with straight/wavy/black/blond/brown/gray hair.
"A photo of {} {} {} that is {}.",      # -> A photo of an attractive young man/woman that is bald.
"A photo of {} {} with {} {} hair.",    # -> A photo of an attractive young man/woman with straight/wavy black/blond/brown/gray hair.
"A photo of {} {} with {} hair.",       # -> A photo of an attractive young man/woman with straight/wavy/black/blond/brown/gray hair.
"A photo of {} {} that is {}.",         # -> A photo of an attractive/a young man/woman that is bald.
]

second_sentence_templates = [
"{} has {} and {}.",                                    # -> He/She has a pale skin and an oval face.
"{} has {}.",                                           # -> He/She has a pale skin/an oval face.

"{} has {} and {} with {}, {}, {}.",                      # -> He/She has a pale skin and an oval face with arched eyebrows/pointy nose/high cheekbones (or any combination of these three attributes)
"{} has {} with {}, {}, {}.",                             # -> He/She has a pale skin/an oval face with arched eyebrows/pointy nose/high cheekbones (or any combination of these three attributes)

"{} has {} and {} with {}, {}, {} and {}.",               # -> He/She has a pale skin and an oval face with arched eyebrows/pointy nose/high cheekbones (or any combination of these three attributes) and narrow eyes
"{} has {} with {}, {}, {} and {}.",                      # -> He/She has a pale skin/an oval face with arched eyebrows/pointy nose/high cheekbones (or any combination of these three attributes) and narrow eyes

"{} has {} and {} with {}, {} ,{} and {} with {}.",       # -> He/She has a pale skin and an oval face with arched eyebrows/pointy nose/high cheekbones (or any combination of these three attributes) and narrow eyes with bags under the eyes
"{} has {} with {}, {}, {} and {} with {}.",              # -> He/She has a pale skin/an oval face with arched eyebrows/pointy nose/high cheekbones (or any combination of these three attributes) and narrow eyes with bags under the eyes

"{} has {} and {} with {} {} {} and also {}.",          # -> He/She has a pale skin and an oval face with arched eyebrows/pointy nose/high cheekbones (or any combination of these three attributes) and also bags under the eyes
"{} has {} with {}, {}, {} and also {}.",                 # -> He/She has a pale skin/an oval face with arched eyebrows/pointy nose/high cheekbones (or any combination of these three attributes) and also bags under the eyes

"{} has {}, {}, {}, {}, {}, {}, {}.",                           # -> He/She has (whichever attribute is provided) -> if the attributes [pale skin, oval face] are not provided
]

third_sentence_templates = [
"{} is {}.",                                        # -> He/She is smiling. (if big lips is not provided)
"{} is {} with {}.",                                # -> He/She is smiling with big lips.

"{} is {} and {} is wearing {}, {}, {}.",                   # -> He/She is smiling and he/she is wearing (whatever combination of heavy makeup, a necklace and a necktie is provided)
"{} is {} with {} and {} is wearing {}, {}, {}.",           # -> He/She is smiling with big lips and he/she is wearing heavy makeup/a necklace/a necktie(whatever combination of heavy makeup, a necklace and a necktie is provided)


"{} has {} and {} wears {}, {}, {}.",                  # -> He/She has big lips and he/she is wearing.. (if smiling is not provided)
"{} wears {}, {}, {}.",                                  # -> He/She wears a necklace, a necktie, a heavy makeup 

"{} has {}."                                            # -> He/She has big lips.
]

fourth_sentence_templates = [
    "{} has {} and {}.",                                # -> He/She has eyeglasses and a beard/no beard.
    "{} has {}."                                        # He/She has eyeglasses/a beard/no beard
]

data_directory = "/mnt/hdd/volume1/anastasija/CelebA/Img/img_celeba/"
eval_directory = "/mnt/hdd/volume1/anastasija/CelebA/Eval/"
annotations_directory = "/mnt/hdd/volume1/anastasija/CelebA/Anno/"

annotations_filename = "list_attr_celeba.txt"
annotations_path = os.path.join(annotations_directory, annotations_filename)

annotations_df = pd.read_csv(annotations_path, delimiter='\s+', skiprows=[0])
attributes = list(annotations_df.columns.values)

list_eval_partition = pd.read_csv(os.path.join(eval_directory, "list_eval_partition.txt"), sep=" ", header=None)
img_filenames_all = sorted(glob.glob(data_directory + '*.jpg'))


chosen_attributes = [
    "Male",
    "Straight_Hair",
    "Wavy_Hair",
    "Bald",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Attractive",
    "Young",
    "Wearing_Hat",
    "Smiling",
    "Eyeglasses",
    "No_Beard",
    "Arched_Eyebrows",
    "Narrow_Eyes",
    "Bags_Under_Eyes",
    "Big_Lips",
    "Oval_Face",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Pale_Skin",
    "Pointy_Nose"
]

chosen_att_dict = {
    "Male": ["man", "woman"],
    "Straight_Hair": "straight",
    "Wavy_Hair": "wavy",
    "Bald": "bald",
    "Black_Hair": "black",
    "Blond_Hair": "blond",
    "Brown_Hair": "brown",
    "Gray_Hair": "gray",
    "Attractive": "an attractive",
    "Young": "a young",
    "Smiling": "smiling",
    "Eyeglasses": "eyeglasses",
    "No_Beard": ["no beard", "a beard"],
    "Arched_Eyebrows": "arched eyebrows",
    "Narrow_Eyes": "narrow eyes",
    "Bags_Under_Eyes": "bags under the eyes",
    "Big_Lips": "big lips",
    "Oval_Face": "an oval face",
    "Wearing_Necklace": "necklace",
    "Wearing_Necktie": "necktie",
    "Heavy_Makeup": "heavy makeup",
    "High_Cheekbones": "high cheekbones",
    "Pale_Skin": "a pale skin",
    "Pointy_Nose": "a pointy nose"
}

first_sent_attr = [
    "Male",
    "Straight_Hair",
    "Wavy_Hair",
    "Bald",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
    "Attractive",
    "Young"
]

second_sent_attr = [
    "Pale_Skin",
    "Oval_Face",
    "Arched_Eyebrows",
    "Pointy_Nose",
    "High_Cheekbones",
    "Narrow_Eyes",
    "Bags_Under_Eyes"
]

third_sent_attr = [
    "Smiling",
    "Big_Lips",
    "Heavy_Makeup",
    "Wearing_Necklace",
    "Wearing_Necktie"
]

fourth_sent_attr = [
    "Eyeglasses",
    "No_Beard"
]

def generate_first_sentence(first_attr, first_sentence_templates):
    if first_attr["Male"] > 0:
        gender = "man"
    elif first_attr["Male"] < 0:
        gender = "woman"

    if first_attr["Bald"] > 0:
        bald = "bald"
        hair_type = ""
        hair_color = ""
    elif first_attr["Bald"] < 0:
        bald = ""
        if first_attr["Straight_Hair"] > 0:
            hair_type = "straight"
        elif first_attr["Wavy_Hair"] > 0:
            hair_type = "wavy"
        else:
            hair_type = ""
        
        if first_attr["Black_Hair"] > 0:
            hair_color = "black"
        elif first_attr["Blond_Hair"] > 0:
            hair_color = "blond"
        elif first_attr["Brown_Hair"] > 0:
            hair_color = "brown"
        elif first_attr["Gray_Hair"] > 0:
            hair_color = "gray"
        else:
            hair_color = ""
    
    if first_attr["Attractive"] > 0:
        attractive = "an attractive"
    else:
        attractive = ""
    if first_attr ["Young"] > 0:
        young = "a young"
    else:
        young = ""
    
    if not attractive and not young and not hair_color and not hair_type and not bald:
        template = first_sentence_templates[0]
        first_sentence = template.format(gender)
    elif attractive and young and not hair_color and not hair_type and not bald:
        template = first_sentence_templates[1]
        first_sentence = template.format(attractive, young[2:], gender)
    elif attractive and not young and not hair_color and not hair_type and not bald:
        template = first_sentence_templates[2]
        first_sentence = template.format(attractive, gender)
    elif not attractive and young and not hair_color and not hair_type and not bald:
        template = first_sentence_templates[2]
        first_sentence = template.format(young, gender)

    elif not attractive and not young and hair_color and hair_type and not bald:
        template = first_sentence_templates[3]
        first_sentence = template.format(gender, hair_type, hair_color)
    elif not attractive and not young and hair_color and not hair_type and not bald:
        template = first_sentence_templates[4]
        first_sentence = template.format(gender, hair_color)
    elif not attractive and not young and hair_type and not hair_color and not bald:
        template = first_sentence_templates[4]
        first_sentence = template.format(gender, hair_type)
    elif not attractive and not young and not hair_color and not hair_type and bald:
        template = first_sentence_templates[5]
        first_sentence = template.format(gender, bald)

    elif attractive and young and hair_type and hair_color and not bald:
        template = first_sentence_templates[6]
        first_sentence = template.format(attractive, young[2:], gender, hair_type, hair_color)
    elif attractive and young and hair_type and not hair_color and not bald:
        template = first_sentence_templates[7]
        first_sentence = template.format(attractive, young[2:], gender, hair_type)
    elif attractive and young and not hair_type and hair_color and not bald:
        template = first_sentence_templates[7]
        first_sentence = template.format(attractive, young[2:], gender, hair_color)
    elif attractive and young and not hair_type and not hair_color and bald:
        template = first_sentence_templates[8]
        first_sentence = template.format(attractive, young, gender, bald)

    elif attractive and not young and hair_type and hair_color and not bald:
        template = first_sentence_templates[9]
        first_sentence = template.format(attractive, gender, hair_type, hair_color)
    elif attractive and not young and hair_type and not hair_color and not bald:
        template = first_sentence_templates[10]
        first_sentence = template.format(attractive, gender, hair_type)
    elif attractive and not young and not hair_type and hair_color and not bald:
        template = first_sentence_templates[10]
        first_sentence = template.format(attractive, gender, hair_color)
    elif attractive and not young and not hair_type and not hair_color and bald:
        template = first_sentence_templates[11]
        first_sentence = template.format(attractive, gender, bald)
    
    elif not attractive and young and hair_type and hair_color and not bald:
        template = first_sentence_templates[9]
        first_sentence = template.format(young, gender, hair_type, hair_color)
    elif not attractive and young and hair_type and not hair_color and not bald:
        template = first_sentence_templates[10]
        first_sentence = template.format(young, gender, hair_type)
    elif not attractive and young and not hair_type and hair_color and not bald:
        template = first_sentence_templates[10]
        first_sentence = template.format(young, gender, hair_color)
    elif not attractive and young and not hair_type and not hair_color and bald:
        template = first_sentence_templates[11]
        first_sentence = template.format(young, gender, bald)

    return first_sentence

def generate_second_sentence(pronoun, second_attr, second_sentence_templates):
    if second_attr["Arched_Eyebrows"] > 0:
        arched_eyebrows = "arched eyebrows"
    elif second_attr["Arched_Eyebrows"] < 0:
        arched_eyebrows = ""
    
    if second_attr["Narrow_Eyes"] > 0:
        narrow_eyes = "narrow eyes"
    elif second_attr["Narrow_Eyes"] < 0:
        narrow_eyes = ""
    
    if second_attr["Bags_Under_Eyes"] > 0:
        bags_under_eyes = "bags under the eyes"
    elif second_attr["Bags_Under_Eyes"] < 0:
        bags_under_eyes = ""

    if second_attr["Oval_Face"] > 0:
        oval_face = "an oval face"
    elif second_attr["Oval_Face"] < 0:
        oval_face = ""
    
    if second_attr["High_Cheekbones"] > 0:
        high_cheekbones = "high cheekbones"
    elif second_attr["High_Cheekbones"] < 0:
        high_cheekbones = ""
    
    if second_attr["Pale_Skin"] > 0:
        pale_skin = "a pale skin"
    elif second_attr["Pale_Skin"] < 0:
        pale_skin = ""
    
    if second_attr["Pointy_Nose"] > 0:
        pointy_nose = "a pointy nose"
    elif second_attr["Pointy_Nose"] < 0:
        pointy_nose = ""
    
    if pale_skin and oval_face and not arched_eyebrows and not narrow_eyes and not bags_under_eyes and not high_cheekbones and not pointy_nose:
        template = second_sentence_templates[0]
        second_sentence = template.format(pronoun, pale_skin, oval_face)
    elif pale_skin and not oval_face and not arched_eyebrows and not narrow_eyes and not bags_under_eyes and not high_cheekbones and not pointy_nose:
        template = second_sentence_templates[1]
        second_sentence = template.format(pronoun, pale_skin)
    elif not pale_skin and oval_face and not arched_eyebrows and not narrow_eyes and not bags_under_eyes and not high_cheekbones and not pointy_nose:
        template = second_sentence_templates[1]
        second_sentence = template.format(pronoun, oval_face)
    
    elif pale_skin and oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and not narrow_eyes and not bags_under_eyes:
        template = second_sentence_templates[2]
        second_sentence = template.format(pronoun, pale_skin, oval_face, arched_eyebrows, pointy_nose, high_cheekbones)
    elif pale_skin and not oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and not narrow_eyes and not bags_under_eyes:
        template = second_sentence_templates[3]
        second_sentence = template.format(pronoun, pale_skin, arched_eyebrows, pointy_nose, high_cheekbones)
    elif not pale_skin and oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and not narrow_eyes and not bags_under_eyes:
        template = second_sentence_templates[3]
        second_sentence = template.format(pronoun, oval_face, arched_eyebrows, pointy_nose, high_cheekbones)
    
    elif pale_skin and oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and narrow_eyes and not bags_under_eyes:
        template = second_sentence_templates[4]
        second_sentence = template.format(pronoun, pale_skin, oval_face, arched_eyebrows, pointy_nose, high_cheekbones, narrow_eyes)
    elif pale_skin and not oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and narrow_eyes and not bags_under_eyes:
        template = second_sentence_templates[5]
        second_sentence = template.format(pronoun, pale_skin, arched_eyebrows, pointy_nose, high_cheekbones, narrow_eyes)
    elif not pale_skin and oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and narrow_eyes and not bags_under_eyes:
        template = second_sentence_templates[5]
        second_sentence = template.format(pronoun, oval_face, arched_eyebrows, pointy_nose, high_cheekbones, narrow_eyes)
    
    elif pale_skin and oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and narrow_eyes and bags_under_eyes:
        template = second_sentence_templates[6]
        second_sentence = template.format(pronoun, pale_skin, oval_face, arched_eyebrows, pointy_nose, high_cheekbones, narrow_eyes, bags_under_eyes)
    elif pale_skin and not oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and narrow_eyes and bags_under_eyes:
        template = second_sentence_templates[7]
        second_sentence = template.format(pronoun, pale_skin, arched_eyebrows, pointy_nose, high_cheekbones, narrow_eyes, bags_under_eyes)
    elif not pale_skin and oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and narrow_eyes and bags_under_eyes:
        template = second_sentence_templates[7]
        second_sentence = template.format(pronoun, pale_skin, oval_face, arched_eyebrows, pointy_nose, high_cheekbones, narrow_eyes, bags_under_eyes)
    
    elif pale_skin and oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and not narrow_eyes and bags_under_eyes:
        template = second_sentence_templates[8]
        second_sentence = template.format(pronoun, pale_skin, oval_face, arched_eyebrows, pointy_nose, high_cheekbones, bags_under_eyes)
    elif pale_skin and not oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and not narrow_eyes and bags_under_eyes:
        template = second_sentence_templates[9]
        second_sentence = template.format(pronoun, pale_skin, arched_eyebrows, pointy_nose, high_cheekbones, bags_under_eyes)
    elif not pale_skin and oval_face and (arched_eyebrows or pointy_nose or high_cheekbones) and not narrow_eyes and bags_under_eyes:
        template = second_sentence_templates[9]
        second_sentence = template.format(pronoun, oval_face, arched_eyebrows, pointy_nose, high_cheekbones, bags_under_eyes)

    # elif not pale_skin and not oval_face and (arched_eyebrows or pointy_nose or high_cheekbones or narrow_eyes or bags_under_eyes):
    #     template = second_sentence_templates[10]
    #     second_sentence = template.format(pronoun, arched_eyebrows, pointy_nose, high_cheekbones, narrow_eyes, bags_under_eyes)
    elif not (pale_skin and oval_face and arched_eyebrows and pointy_nose and high_cheekbones and narrow_eyes) and bags_under_eyes:
        template = second_sentence_templates[1]
        second_sentence = template.format(pronoun, bags_under_eyes)


    elif (pale_skin or oval_face or arched_eyebrows or pointy_nose or high_cheekbones or narrow_eyes or bags_under_eyes):
        template = second_sentence_templates[10]
        second_sentence = template.format(pronoun, pale_skin, oval_face, arched_eyebrows, pointy_nose, high_cheekbones, narrow_eyes, bags_under_eyes)

    second_sentence = re.sub(r'\s+', ' ', second_sentence)
    second_sentence = re.sub(r' ,+', '', second_sentence)
    second_sentence = re.sub(r', *\.', '.', second_sentence)

    return second_sentence

def generate_third_sentence(pronoun, third_attr, third_sentence_templates):
    if third_attr["Smiling"] > 0:
        smiling = "smiling"
    elif third_attr["Smiling"] < 0:
        smiling = ""

    if third_attr["Big_Lips"] > 0:
        big_lips = "big lips"
    elif third_attr["Big_Lips"] < 0:
        big_lips = ""
    
    if third_attr["Wearing_Necklace"] > 0:
        necklace = "a necklace"
    elif third_attr["Wearing_Necklace"] < 0:
        necklace = ""
    
    if third_attr["Wearing_Necktie"] > 0:
        necktie = "a necktie"
    elif third_attr["Wearing_Necktie"] < 0:
        necktie = ""
    
    if third_attr["Heavy_Makeup"] > 0:
        makeup = "a heavy makeup"
    elif third_attr["Heavy_Makeup"] < 0:
        makeup = ""
    
    if smiling and not big_lips and not necklace and not necktie and not makeup:
        template = third_sentence_templates[0]
        third_sentence = template.format(pronoun, smiling)
    elif smiling and big_lips and not necklace and not necktie and not makeup:
        template = third_sentence_templates[1]
        third_sentence = template.format(pronoun, smiling, big_lips)
    
    elif smiling and not big_lips and (necklace or necktie or makeup):
        template = third_sentence_templates[2]
        third_sentence = template.format(pronoun, smiling, pronoun.lower(), necklace, necktie, makeup)
    elif smiling and big_lips and (necklace or necktie or makeup):
        template = third_sentence_templates[3]
        third_sentence = template.format(pronoun, smiling, big_lips, pronoun.lower(), necklace, necktie, makeup)
    
    elif not smiling and big_lips and not necklace and not necktie and not makeup:
        template = third_sentence_templates[-1]
        third_sentence = template.format(pronoun, big_lips)
    elif not smiling and not big_lips and (necklace or necktie or makeup):
        template = third_sentence_templates[5]
        third_sentence = template.format(pronoun, necklace, necktie, makeup)
    elif not smiling and big_lips and (necklace or necktie or makeup):
        template = third_sentence_templates[4]
        third_sentence = template.format(pronoun, big_lips, pronoun.lower(), necklace, necktie, makeup)
    
    third_sentence = re.sub(r'\s+', ' ', third_sentence)
    third_sentence = re.sub(r' ,+', '', third_sentence)
    third_sentence = re.sub(r', *\.', '.', third_sentence)

    return third_sentence

def generate_fourth_sentence(pronoun, fourth_attr, fourth_sentence_templates):
    if fourth_attr["Eyeglasses"] > 0:
        eyeglasses = "eyeglasses"
    elif fourth_attr["Eyeglasses"] < 0:
        eyeglasses = ""
    
    if fourth_attr["No_Beard"] > 0:
        no_beard = "no beard"
    elif fourth_attr["No_Beard"] < 0:
        no_beard = ""

    if eyeglasses and no_beard:
        template = fourth_sentence_templates[0]
        fourth_sentence = template.format(pronoun, eyeglasses, no_beard)
    elif eyeglasses and not no_beard:
        template = fourth_sentence_templates[1]
        fourth_sentence = template.format(pronoun, "a beard")
    elif not eyeglasses and no_beard:
        template = fourth_sentence_templates[1]
        fourth_sentence = template.format(pronoun, no_beard)

    return fourth_sentence


test_images = img_filenames_all
captions = {}
att_dict = {}
for image in tqdm(test_images):
    attribute_dict = {}
    image_name = os.path.basename(image)
    attributes = annotations_df.loc[image_name, chosen_attributes]
    attributes_names = list(attributes.keys())
    attributes_names.sort(key = lambda i: chosen_attributes.index(i))

    first_attr = attributes[attributes.keys().isin(first_sent_attr)]
    second_attr = attributes[attributes.keys().isin(second_sent_attr)]
    third_attr = attributes[attributes.keys().isin(third_sent_attr)]
    fourth_attr = attributes[attributes.keys().isin(fourth_sent_attr)]

    first_sentence = generate_first_sentence(first_attr, first_sentence_templates)

    if attributes["Male"] > 0:
        pronoun = "He"
    elif attributes["Male"] < 0:
        pronoun = "She"
    
    if np.any(second_attr.values > 0):
        second_sentence = generate_second_sentence(pronoun, second_attr, second_sentence_templates)
    else:
        second_sentence = ""

    if np.any(third_attr.values > 0):
        third_sentence = generate_third_sentence(pronoun, third_attr, third_sentence_templates)
    else:
        third_sentence = ""
    
    if np.any(fourth_attr.values > 0):
        fourth_sentence = generate_fourth_sentence(pronoun, fourth_attr, fourth_sentence_templates)
    else:
        fourth_sentence = ""
    
    all_sentences = first_sentence + ' ' + second_sentence + ' ' + third_sentence + ' ' + fourth_sentence
    captions[image_name] = re.sub(r'\s+', ' ', all_sentences)

# Save the captions
output_file = 'data/captions/captions_19_attr_all_images.txt'

# Open the file in write mode and save the data
with open(output_file, 'w') as file:
    for image_name, caption in captions.items():
        file.write(f'{image_name} {caption}\n')
