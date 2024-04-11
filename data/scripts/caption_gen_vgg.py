import random

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

attribute_captions = {
    'Male': ['man', 'gentleman', 'male'],
    'Young': ['youthful', 'young'],
    'Middle_Aged': ['middle-aged', 'prime-aged'],
    'Senior': ['elderly', 'senior'],
    'Asian': ['Asian', 'Asian-descent'],
    'White': ['Caucasian', 'white-skinned'],
    'Black': ['African American', 'black-skinned'],
    'Rosy_Cheeks': ['rosy-cheeked', 'glowing'],
    'Shiny_Skin': ['radiant-skinned', 'glowing'],
    'Bald': ['bald', 'hairless'],
    'Wavy_Hair': ['wavy-haired', 'curly-haired'],
    'Receding_Hairline': ['receding-haired', 'balding'],
    'Bangs': ['bang-sporting', 'fringe-wearing'],
    'Sideburns': ['sideburn-sporting', 'facial-haired'],
    'Black_Hair': ['dark-haired', 'black-locked'],
    'Blond_Hair': ['blond-haired', 'light-haired'],
    'Brown_Hair': ['brown-haired', 'chestnut-locked'],
    'Gray_Hair': ['gray-haired', 'silver-haired'],
    'No_Beard': ['clean-shaven', 'beardless'],
    'Mustache': ['mustachioed', 'mustache-wearing'],
    '5_o_Clock_Shadow': ['stubbled', 'shadowed'],
    'Goatee': ['goatee-wearing', 'bearded'],
    'Oval_Face': ['oval-faced', 'oval-faced'],
    'Square_Face': ['square-faced', 'broad-jawed'],
    'Round_Face': ['round-faced', 'circular-faced'],
    'Double_Chin': ['double-chinned', 'extra-skinned'],
    'High_Cheekbones': ['high-cheekboned', 'cheekboned'],
    'Chubby': ['chubby-cheeked', 'full-cheeked'],
    'Obstructed_Forehead': ['forehead-obscured', 'partially-covered'],
    'Fully_Visible_Forehead': ['forehead-visible', 'unobscured'],
    'Brown_Eyes': ['brown-eyed', 'brown-eyed'],
    'Bags_Under_Eyes': ['baggy-eyed', 'fatigued-eyed'],
    'Bushy_Eyebrows': ['bushy-browed', 'thick-browed'],
    'Arched_Eyebrows': ['arched-browed', 'gracefully-arched'],
    'Mouth_Closed': ['closed-mouthed', 'closed-lipped'],
    'Smiling': ['smiling', 'joyful'],
    'Big_Lips': ['full-lipped', 'plump-lipped'],
    'Big_Nose': ['prominent-nosed', 'large-nosed'],
    'Pointy_Nose': ['pointed-nosed', 'sharp-nosed'],
    'Heavy_Makeup': ['heavily-made-up', 'noticeable-makeup'],
    'Wearing_Hat': ['hat-wearing', 'headwear-sporting'],
    'Wearing_Earrings': ['earring-wearing', 'ear-accessorized'],
    'Wearing_Necktie': ['necktie-wearing', 'formally-dressed'],
    'Wearing_Lipstick': ['lipstick-wearing', 'lip-colored'],
    'No_Eyewear': ['eyewear-less', 'glasses-less'],
    'Eyeglasses': ['glasses-wearing', 'spectacled'],
    'Attractive': ['attractive', 'charming', 'handsome']
}


def generate_caption(attributes, attribute_captions, groups):
    caption = "A"
    for group, attributes_list in groups.items():
        group_caption = ""
        for attribute in attributes_list:
            if attributes.get(attribute) == 1:
                attribute_caption = random.choice(attribute_captions.get(attribute, []))
                if attribute_caption:
                    group_caption += " " + attribute_caption
        if group_caption:
            caption += group_caption + ","
    
    return caption[:-1] + " individual."

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

caption = generate_caption(image_attributes, attribute_captions, groups)
print(caption)

##################################################################################
##################################################################################

def describe_face(attributes):
    parts = []

    # Gender and age
    if attributes['Male'] == 1:
        gender = 'man'
    else:
        gender = 'woman'

    age_groups = []
    if attributes['Young'] == 1:
        age_groups.append('young')
    if attributes['Middle_Aged'] == 1:
        age_groups.append('middle-aged')
    if attributes['Senior'] == 1:
        age_groups.append('senior')
    
    age_description = ' '.join(age_groups) if age_groups else 'person'
    parts.append(f"A photo of a {age_description} {gender}")

    # Ethnicity
    ethnicities = []
    if attributes['Asian'] == 1:
        ethnicities.append('asian')
    if attributes['White'] == 1:
        ethnicities.append('white')
    if attributes['Black'] == 1:
        ethnicities.append('black')
    if ethnicities:
        parts.append(f"of {'/'.join(ethnicities)} descent")

    # Hair and facial features
    hair_features = []
    if attributes['Bald'] == 1:
        parts.append(' that is bald')
    else:
        if attributes['Black_Hair'] == 1:
            hair_features.append('with black hair')
        if attributes['Blond_Hair'] == 1:
            hair_features.append('with blond hair')
        if attributes['Brown_Hair'] == 1:
            hair_features.append('with brown hair')
        if attributes['Gray_Hair'] == 1:
            hair_features.append('with gray hair')
        if attributes['Wavy_Hair'] == 1:
            hair_features.append('that is wavy')
        if attributes['Receding_Hairline'] == 1:
            hair_features.append('with a receding hairline')
    
    if hair_features:
        parts.append(', '.join(hair_features))

    # Face attributes
    face_features = []
    if attributes['No_Beard'] == -1:
        if attributes['Mustache'] == 1:
            face_features.append('a mustache')
        if attributes['5_o_Clock_Shadow'] == 1:
            face_features.append('a 5 o\'clock shadow')
        if attributes['Goatee'] == 1:
            face_features.append('a goatee')

    if attributes['Double_Chin'] == 1:
        hair_and_face_features.append('a double chin')

    if hair_and_face_features:
        parts.append(', '.join(hair_and_face_features))

    # Accessories and other features
    accessories = []
    if attributes['Wearing_Hat'] == 1:
        accessories.append('wearing a hat')
    if attributes['Wearing_Necktie'] == 1:
        accessories.append('wearing a necktie')
    if attributes['Heavy_Makeup'] == 1:
        accessories.append('wearing heavy makeup')
    if accessories:
        parts.append(', '.join(accessories))

    return '. '.join(parts) + '.'

# Example usage
attributes = {
    'Male': 1, 'Young': -1, 'Middle_Aged': 1, 'Senior': -1, 'Asian': -1, 'White': 1, 'Black': -1,
    'Bald': -1, 'Wavy_Hair': -1, 'Receding_Hairline': -1, 'Black_Hair': 1, 'Blond_Hair': -1,
    'Brown_Hair': -1, 'Gray_Hair': -1, 'No_Beard': -1, 'Mustache': 1, '5_o_Clock_Shadow': -1,
    'Goatee': -1, 'Double_Chin': 1, 'Wearing_Hat': -1, 'Wearing_Necktie': 1, 'Heavy_Makeup': -1
}

print(describe_face(attributes))

