import os
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""     DISTRIBUTION ANALYSIS     """

def get_annotation(ann_path, columns=None, verbose=True):
    df = pd.read_csv(ann_path)

    return df

# Whole dataset
annotations_path = '/mnt/hdd/volume1/MAAD-Face'
attr = get_annotation(os.path.join(annotations_path, 'MAAD_Face.csv'), verbose=False)

columns = []
for index, attribute_name in enumerate(attr.columns[2:]):
    columns.append(attribute_name)
    #print(f"{index}: {attribute_name}")

# Collect the results into a dictionary
results = {}
for colnm in columns:
    results[colnm] = 100 * np.mean(attr[colnm] == 1)

# Sort the dictionary by values
sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

# Print the sorted items
for colnm, value in sorted_results.items():
    print(" {:20} {:5.2f}%".format(colnm, value))

frequencies_positive = (attr.iloc[:,2:] == 1).mean(axis=0)

# _ = frequencies_positive.plot(title='VGGFace2 Positive Attributes', 
#                      kind='bar', figsize=(12, 5), color='g')

frequencies_negative = (attr.iloc[:,2:] == -1).mean(axis=0)

frequencies_neutral = (attr.iloc[:,2:] == 0).mean(axis=0)

# Define the positions for each group of bars
bar_width = 0.25
p1 = range(len(frequencies_positive))
p2 = [x + bar_width for x in p1]
p3 = [x + bar_width for x in p2]

# Plot the bars
plt.figure(figsize=(20, 8))
plt.bar(p1, frequencies_positive.values, color='g', width=bar_width, edgecolor='grey', label='Positive')
plt.bar(p2, frequencies_negative.values, color='r', width=bar_width, edgecolor='grey', label='Negative')
plt.bar(p3, frequencies_neutral.values, color='y', width=bar_width, edgecolor='grey', label='Neutral')

# Add labels and title
plt.xlabel('Attributes', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('VGG Face 2 Face Attributes', fontsize=16)
plt.xticks([r + bar_width for r in range(len(frequencies_positive))], frequencies_positive.index, rotation=90)
plt.legend()

plt.show()

# Stacked bar
plt.figure(figsize=(20, 8))
plt.bar(columns, frequencies_positive.values, color='g', width=bar_width, edgecolor='grey', label='Positive')
plt.bar(columns, frequencies_negative.values, bottom=frequencies_positive.values, color='r', width=bar_width, edgecolor='grey', label='Negative')
plt.bar(columns, frequencies_neutral.values, bottom=frequencies_positive.values+frequencies_negative.values, color='y', width=bar_width, edgecolor='grey', label='Neutral')

# Add labels and title
plt.xlabel('Attributes', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('VGG Face 2 Face Attributes', fontsize=16)
plt.xticks([r + bar_width for r in range(len(frequencies_positive))], frequencies_positive.index, rotation=90)
plt.legend()

plt.show()

"""     CORRELATION MATRIX     """
fig, ax = plt.subplots(figsize=(15,15))
attr.loc[:, columns] = attr.loc[:, columns].replace(0, -1)
attr_corr = attr.loc[:, columns].corr()
sns.heatmap(attr_corr, cmap="RdYlBu", vmin=-1, vmax=1, cbar=True)
plt.show()

corr = attr.loc[:, columns].corr()
corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool_))
corr = corr.unstack().transpose().sort_values(ascending=False).dropna()

pd.set_option('display.max_rows', 1100)
corr



