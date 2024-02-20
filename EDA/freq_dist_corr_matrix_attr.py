import os
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""     DISTRIBUTION ANALYSIS     """

def get_annotation(fnmtxt, columns=None, verbose=True):
    if verbose:
        print("_"*70)
        print(fnmtxt)
    
    rfile = open(fnmtxt, 'r' ) 
    texts = rfile.readlines()
    rfile.close()
    
    if not columns:
        columns = np.array(texts[1].split(" "))
        columns = columns[columns != "\n"]
        texts = texts[2:]
    
    df = []
    for txt in texts:
        txt = np.array(txt.rstrip("\n").split(" "))
        txt = txt[txt != ""]
    
        df.append(txt)
        
    df = pd.DataFrame(df)

    if df.shape[1] == len(columns) + 1:
        columns = ["image_id"]+ list(columns)
    df.columns = columns   
    df = df.dropna()
    if verbose:
        print(" Total number of annotations {}\n".format(df.shape))
        print(df.head())
    ## cast to integer
    for nm in df.columns:
        if nm != "image_id":
            df[nm] = pd.to_numeric(df[nm],downcast="integer")
    return(df)

# Whole dataset
annotations_path = '/mnt/hdd/volume1/anastasija/CelebA/Anno'
attr = get_annotation(os.path.join(annotations_path, 'list_attr_celeba.txt'), verbose=False)

columns = []
for index, attribute_name in enumerate(attr.columns[1:]):
    columns.append(attribute_name)
    #print(f"{index}: {attribute_name}")

for colnm in attr.columns:
    if colnm != "image_id":
        print(" {:20} {:5.2f}%".format(colnm,100*np.mean(attr[colnm] == 1)))

frequencies = (attr.iloc[:,1:] == 1).mean(axis=0).sort_values()
print(len(frequencies))

_ = frequencies.plot(title='CelebA Face Attributes', 
                     kind='bar', figsize=(12, 5), color='g')

# For every partition
train_attr = attr.iloc[:162770]
val_attr = attr.iloc[162770:182637]
test_attr = attr.iloc[182637:]
stat_df = pd.DataFrame(index = columns) #, columns = ['Full','Train','Val','Test'])
stat_df.loc[:,'Full'] = (attr.iloc[:,1:] == 1).mean(axis=0)
stat_df.loc[:,'Train'] = (train_attr.iloc[:,1:] == 1).mean(axis=0)
stat_df.loc[:,'Val'] = (val_attr.iloc[:,1:] == 1).mean(axis=0)
stat_df.loc[:,'Test'] = (test_attr.iloc[:,1:] == 1).mean(axis=0)

with sns.axes_style('white'):
    stat_df = stat_df.sort_values('Full', ascending=False)
    stat_df.plot(title='CelebA Face Attributes Frequency Distribution', 
                 kind='bar', figsize=(20, 5))
    # plt.savefig('freq_dis.jpg', dpi=160, bbox_inches='tight')


"""     CORRELATION MATRIX     """
fig, ax = plt.subplots(figsize=(15,15))
attr_corr = attr.loc[:, attr.columns != 'image_id'].corr()
sns.heatmap(attr_corr, cmap="RdYlBu", vmin=-1, vmax=1, cbar=False)
plt.show()

corr = attr.loc[:, attr.columns != 'image_id'].corr()
corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))
corr = corr.unstack().transpose().sort_values(ascending=False).dropna()

corr



