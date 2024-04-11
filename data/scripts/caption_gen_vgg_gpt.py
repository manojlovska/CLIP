import random

# Define attribute labels
attribute_captions = {
    'Male': ['man', 'gentleman', 'male individual'],
    'Young': ['youthful individual', 'young person'],
    'Middle_Aged': ['middle-aged person', 'person in their prime'],
    'Senior': ['elderly individual', 'senior person'],
    'Asian': ['person of Asian descent', 'individual with Asian heritage'],
    'White': ['Caucasian person', 'person with white skin'],
    'Black': ['African American individual', 'person with black skin'],
    'Rosy_Cheeks': ['person with rosy cheeks', 'individual with a healthy glow'],
    'Shiny_Skin': ['person with radiant skin', 'individual with a glowing complexion'],
    'Bald': ['bald individual', 'person without any hair'],
    'Wavy_Hair': ['person with wavy locks', 'individual with curly hair'],
    'Receding_Hairline': ['individual with a receding hairline', 'person showing signs of baldness'],
    'Bangs': ['person with stylish bangs', 'individual sporting fringe'],
    'Sideburns': ['person with sideburns', 'individual having facial hair along the sides'],
    'Black_Hair': ['person with dark hair', 'individual having black locks'],
    'Blond_Hair': ['person with blond hair', 'individual with light-colored hair'],
    'Brown_Hair': ['person with brown hair', 'individual having chestnut locks'],
    'Gray_Hair': ['person with gray hair', 'individual with silver hair'],
    'No_Beard': ['clean-shaven individual', 'person without any facial hair'],
    'Mustache': ['person with a mustache', 'individual sporting a mustache'],
    '5_o_Clock_Shadow': ['individual with a hint of stubble', 'person showing a shadow of beard growth'],
    'Goatee': ['person with a goatee', 'individual having a small beard under the chin'],
    'Oval_Face': ['person with an oval-shaped face', 'individual having an oval face structure'],
    'Square_Face': ['individual with a square-shaped face', 'person having a broad jawline'],
    'Round_Face': ['individual with a round face', 'person having a circular face structure'],
    'Double_Chin': ['person with a double chin', 'individual having extra skin under the chin'],
    'High_Cheekbones': ['person with prominent cheekbones', 'individual having high cheekbones'],
    'Chubby': ['chubby-faced person', 'individual having full cheeks'],
    'Obstructed_Forehead': ['individual with the forehead partly covered', 'person with the forehead obscured'],
    'Fully_Visible_Forehead': ['individual with the forehead clearly visible', 'person with the forehead unobscured'],
    'Brown_Eyes': ['person with brown eyes', 'individual having brown-colored eyes'],
    'Bags_Under_Eyes': ['individual with bags under the eyes', 'person showing signs of fatigue under the eyes'],
    'Bushy_Eyebrows': ['person with bushy eyebrows', 'individual having thick eyebrows'],
    'Arched_Eyebrows': ['person with arched eyebrows', 'individual having gracefully arched brows'],
    'Mouth_Closed': ['individual with closed mouth', 'person having closed lips'],
    'Smiling': ['smiling individual', 'person with a joyful expression'],
    'Big_Lips': ['individual with full lips', 'person having plump lips'],
    'Big_Nose': ['person with a prominent nose', 'individual having a large nose'],
    'Pointy_Nose': ['person with a pointed nose', 'individual having a sharp nose'],
    'Heavy_Makeup': ['person with heavy makeup', 'individual wearing noticeable makeup'],
    'Wearing_Hat': ['individual wearing a hat', 'person sporting headwear'],
    'Wearing_Earrings': ['person wearing earrings', 'individual having ear accessories'],
    'Wearing_Necktie': ['individual wearing a necktie', 'person sporting formal attire'],
    'Wearing_Lipstick': ['person wearing lipstick', 'individual with colored lips'],
    'No_Eyewear': ['individual without eyewear', 'person not wearing glasses'],
    'Eyeglasses': ['person wearing glasses', 'individual with spectacles'],
    'Attractive': ['attractive individual', 'charming person', 'handsome individual']
}

def generate_caption(attributes):
    positive_attributes = []
    for attribute, value in attributes.items():
        if value == 1:
            positive_attributes.append(attribute)

    caption = "A "
    if "Young" in positive_attributes:
        caption += "youthful "
    caption += "individual with"
    if "Wavy_Hair" in positive_attributes:
        caption += " wavy locks"
    if "Bangs" in positive_attributes:
        caption += " and stylish bangs"
    if "Brown_Hair" in positive_attributes:
        caption += ", sporting chestnut locks"
    if "Mustache" in positive_attributes:
        caption += ", and a distinguished mustache"
    if "Oval_Face" in positive_attributes:
        caption += ". Their face features an oval structure"
    if "High_Cheekbones" in positive_attributes:
        caption += ", prominent cheekbones,"
    if "Bushy_Eyebrows" in positive_attributes:
        caption += " and thick, gracefully arched eyebrows."
    if "Smiling" in positive_attributes:
        caption += " They express joy with a radiant smile,"
    if "Big_Nose" in positive_attributes:
        caption += " complemented by a prominent nose,"
    if "Heavy_Makeup" in positive_attributes:
        caption += " and noticeable makeup."
    if "Wearing_Earrings" in positive_attributes:
        caption += " They wear elegant ear accessories"
    if "Wearing_Necktie" in positive_attributes:
        caption += " and formal attire,"
    if "Attractive" in positive_attributes:
        caption += " giving off a charming vibe."
    else:
        caption += "."

    return caption

# Example usage:
image_attributes = {
    'Identity': 2,
    'Male': -1,
    'Young': 1,
    'Middle_Aged': -1,
    'Senior': -1,
    'Asian': -1,
    'White': 1,
    'Black': -1,
    'Rosy_Cheeks': 0,
    'Shiny_Skin': 0,
    'Bald': -1,
    'Wavy_Hair': 1,
    'Receding_Hairline': -1,
    'Bangs': 1,
    'Sideburns': -1,
    'Black_Hair': 0,
    'Blond_Hair': 0,
    'Brown_Hair': 1,
    'Gray_Hair': 0,
    'No_Beard': 0,
    'Mustache': 1,
    '5_o_Clock_Shadow': -1,
    'Goatee': 0,
    'Oval_Face': 1,
    'Square_Face': -1,
    'Round_Face': 0,
    'Double_Chin': -1,
    'High_Cheekbones': 1,
    'Chubby': -1,
    'Obstructed_Forehead': 0,
    'Fully_Visible_Forehead': 0,
    'Brown_Eyes': 0,
    'Bags_Under_Eyes': -1,
    'Bushy_Eyebrows': 1,
    'Arched_Eyebrows': 1,
    'Mouth_Closed': -1,
    'Smiling': 1,
    'Big_Lips': -1,
    'Big_Nose': 1,
    'Pointy_Nose': -1,
    'Heavy_Makeup': 1,
    'Wearing_Hat': 0,
    'Wearing_Earrings': 1,
    'Wearing_Necktie': 1,
    'Wearing_Lipstick': -1,
    'No_Eyewear': 1,
    'Eyeglasses': -1,
    'Attractive': 1
}

caption = generate_caption(image_attributes)
print(caption)

#################################################################################################################################################
#################################################################################################################################################

import random

def generate_caption(attributes):
    attribute_groups = {
        'Identity': ['Male', 'Young', 'Middle_Aged', 'Senior'],
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

    caption_parts = []

    for group, group_attributes in attribute_groups.items():
        group_caption_parts = []
        for attribute in group_attributes:
            if attributes.get(attribute) == 1:
                group_caption_parts.append(attribute)
        if group_caption_parts:
            caption_parts.append((group, group_caption_parts))

    caption = "A"
    for group, group_caption_parts in caption_parts:
        if group == 'Identity':
            caption += " " + random.choice(group_caption_parts).lower()
        else:
            caption += " with"
            if group == 'Appearance':
                if 'Wavy_Hair' in group_caption_parts:
                    caption += " wavy locks"
                if 'Bangs' in group_caption_parts:
                    caption += " stylish bangs"
                if 'Brown_Hair' in group_caption_parts:
                    caption += ", sporting chestnut locks"
                if 'Mustache' in group_caption_parts:
                    caption += ", and a distinguished mustache"
                if 'Oval_Face' in group_caption_parts:
                    caption += ". Their face features an oval structure"
                if 'High_Cheekbones' in group_caption_parts:
                    caption += ", prominent cheekbones,"
                if 'Bushy_Eyebrows' in group_caption_parts:
                    caption += " and thick, gracefully arched eyebrows."
                if 'Smiling' in group_caption_parts:
                    caption += " They express joy with a radiant smile,"
                if 'Big_Nose' in group_caption_parts:
                    caption += " complemented by a prominent nose,"
                if 'Heavy_Makeup' in group_caption_parts:
                    caption += " and noticeable makeup."
            elif group == 'Accessories':
                if 'Wearing_Earrings' in group_caption_parts:
                    caption += " elegant ear accessories"
                if 'Wearing_Necktie' in group_caption_parts:
                    caption += " and formal attire,"
            elif group == 'Attractiveness':
                caption += " giving off a charming vibe."

    return caption.capitalize() + "."


# Example usage:
image_attributes = {
    'Identity': 2,
    'Male': -1,
    'Young': 1,
    'Middle_Aged': -1,
    'Senior': -1,
    'Asian': -1,
    'White': 1,
    'Black': -1,
    'Rosy_Cheeks': 0,
    'Shiny_Skin': 0,
    'Bald': -1,
    'Wavy_Hair': 1,
    'Receding_Hairline': -1,
    'Bangs': 1,
    'Sideburns': -1,
    'Black_Hair': 0,
    'Blond_Hair': 0,
    'Brown_Hair': 1,
    'Gray_Hair': 0,
    'No_Beard': 0,
    'Mustache': 1,
    '5_o_Clock_Shadow': -1,
    'Goatee': 0,
    'Oval_Face': 1,
    'Square_Face': -1,
    'Round_Face': 0,
    'Double_Chin': -1,
    'High_Cheekbones': 1,
    'Chubby': -1,
    'Obstructed_Forehead': 0,
    'Fully_Visible_Forehead': 0,
    'Brown_Eyes': 0,
    'Bags_Under_Eyes': -1,
    'Bushy_Eyebrows': 1,
    'Arched_Eyebrows': 1,
    'Mouth_Closed': -1,
    'Smiling': 1,
    'Big_Lips': -1,
    'Big_Nose': 1,
    'Pointy_Nose': -1,
    'Heavy_Makeup': 1,
    'Wearing_Hat': 0,
    'Wearing_Earrings': 1,
    'Wearing_Necktie': 1,
    'Wearing_Lipstick': -1,
    'No_Eyewear': 1,
    'Eyeglasses': -1,
    'Attractive': 1
}

caption = generate_caption(image_attributes)
print(caption)


#################################################################################################################################################
#################################################################################################################################################

def generate_caption(attributes):
    caption = "A"
    groups = {
        'Identity': ['Young', 'Middle_Aged', 'Senior', 'Male'],
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
    
    for group, attributes_list in groups.items():
        group_caption = ""
        for attribute in attributes_list:
            if attributes.get(attribute) == 1:
                if attribute == "Young":
                    group_caption += " youthful"
                elif attribute == "Middle_Aged":
                    group_caption += " middle-aged"
                elif attribute == "Senior":
                    group_caption += " senior"
                elif attribute == "Asian":
                    group_caption += "n Asian"
                elif attribute == "White":
                    group_caption += " Caucasian"
                elif attribute == "Black":
                    group_caption += "n African American"
                elif attribute == "Bald":
                    group_caption += " bald"
                elif attribute == "Wavy_Hair":
                    group_caption += " with wavy locks"
                elif attribute == "Receding_Hairline":
                    group_caption += " with a receding hairline"
                elif attribute == "Bangs":
                    group_caption += " with stylish bangs"
                elif attribute == "Sideburns":
                    group_caption += " with sideburns"
                elif attribute == "Black_Hair":
                    group_caption += " with black hair"
                elif attribute == "Blond_Hair":
                    group_caption += " with blond hair"
                elif attribute == "Brown_Hair":
                    group_caption += " with brown hair"
                elif attribute == "Gray_Hair":
                    group_caption += " with gray hair"
                elif attribute == "No_Beard":
                    group_caption += " clean-shaven"
                elif attribute == "Mustache":
                    group_caption += " with a mustache"
                elif attribute == "5_o_Clock_Shadow":
                    group_caption += " with a 5 o'clock shadow"
                elif attribute == "Goatee":
                    group_caption += " with a goatee"
                elif attribute == "Oval_Face":
                    group_caption += " with an oval face"
                elif attribute == "Square_Face":
                    group_caption += " with a square face"
                elif attribute == "Round_Face":
                    group_caption += " with a round face"
                elif attribute == "Double_Chin":
                    group_caption += " with a double chin"
                elif attribute == "High_Cheekbones":
                    group_caption += " with high cheekbones"
                elif attribute == "Chubby":
                    group_caption += " chubby-faced"
                elif attribute == "Obstructed_Forehead":
                    group_caption += " with an obstructed forehead"
                elif attribute == "Fully_Visible_Forehead":
                    group_caption += " with a fully visible forehead"
                elif attribute == "Brown_Eyes":
                    group_caption += " with brown eyes"
                elif attribute == "Bags_Under_Eyes":
                    group_caption += " with bags under the eyes"
                elif attribute == "Bushy_Eyebrows":
                    group_caption += " with bushy eyebrows"
                elif attribute == "Arched_Eyebrows":
                    group_caption += " with arched eyebrows"
                elif attribute == "Mouth_Closed":
                    group_caption += " with a closed mouth"
                elif attribute == "Smiling":
                    group_caption += " smiling"
                elif attribute == "Big_Lips":
                    group_caption += " with big lips"
                elif attribute == "Big_Nose":
                    group_caption += " with a big nose"
                elif attribute == "Pointy_Nose":
                    group_caption += " with a pointy nose"
                elif attribute == "Heavy_Makeup":
                    group_caption += " with heavy makeup"
                elif attribute == "Wearing_Hat":
                    group_caption += " wearing a hat"
                elif attribute == "Wearing_Earrings":
                    group_caption += " wearing earrings"
                elif attribute == "Wearing_Necktie":
                    group_caption += " wearing a necktie"
                elif attribute == "Wearing_Lipstick":
                    group_caption += " wearing lipstick"
                elif attribute == "No_Eyewear":
                    group_caption += " not wearing eyewear"
                elif attribute == "Eyeglasses":
                    group_caption += " wearing eyeglasses"
                elif attribute == "Attractive":
                    group_caption += " attractive"
        if group_caption:
            caption += group_caption + ","
    
    return caption[:-1] + " individual."


# Example usage:
image_attributes = {
    'Identity': 2,
    'Male': -1,
    'Young': 1,
    'Middle_Aged': -1,
    'Senior': -1,
    'Asian': -1,
    'White': 1,
    'Black': -1,
    'Rosy_Cheeks': 0,
    'Shiny_Skin': 0,
    'Bald': -1,
    'Wavy_Hair': 1,
    'Receding_Hairline': -1,
    'Bangs': 1,
    'Sideburns': -1,
    'Black_Hair': 0,
    'Blond_Hair': 0,
    'Brown_Hair': 1,
    'Gray_Hair': 0,
    'No_Beard': 0,
    'Mustache': 1,
    '5_o_Clock_Shadow': -1,
    'Goatee': 0,
    'Oval_Face': 1,
    'Square_Face': -1,
    'Round_Face': 0,
    'Double_Chin': -1,
    'High_Cheekbones': 1,
    'Chubby': -1,
    'Obstructed_Forehead': 0,
    'Fully_Visible_Forehead': 0,
    'Brown_Eyes': 0,
    'Bags_Under_Eyes': -1,
    'Bushy_Eyebrows': 1,
    'Arched_Eyebrows': 1,
    'Mouth_Closed': -1,
    'Smiling': 1,
    'Big_Lips': -1,
    'Big_Nose': 1,
    'Pointy_Nose': -1,
    'Heavy_Makeup': 1,
    'Wearing_Hat': 0,
    'Wearing_Earrings': 1,
    'Wearing_Necktie': 1,
    'Wearing_Lipstick': -1,
    'No_Eyewear': 1,
    'Eyeglasses': -1,
    'Attractive': 1
}

caption = generate_caption(image_attributes)
print(caption)


