import pandas as pd

pd.set_option('display.max_colwidth', None)

all_captions = pd.read_csv("/home/anastasija/Documents/Projects/SBS/CLIP/data/captions_all_attributes_no_male.csv", sep="\t")

only_captions = all_captions["caption"]

#specify path for export
path = r'/home/anastasija/Documents/Projects/SBS/CLIP/data/all_captions_all_attributes.csv'

only_captions.to_csv(path, index=False, header=False)

#export DataFrame to text file
with open(path, 'a') as f:
    df_string = only_captions.to_string(header=False, index=False)
    f.write(df_string)


