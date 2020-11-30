# -*- coding: utf-8 -*-
"""
@author: Rashid Haffadi
"""
# %%
import re
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud, ImageColorGenerator
import os.path

# %% Global Variables
languages = ['Bulgarian', 'Czech', 'Danish', 'German', 'Greek', 'English', 'Spanish', 'Estonian', 
             'Finnish', 'French', 'Hungarian', 'Italian', 'Lithuanian', 'Latvian', 'Dutch', 
             'Polish', 'Portuguese', 'Romanian', 'Slovak', 'Slovenian', 'Swedish']

labels = ['bg','cs','da','de','el','en','es','et','fi','fr',
          'hu','it','lt','lv','nl','pl','pt','ro','sk','sl','sv']
# %%


# plt.xticks(rotation=50)

def char_freq(text):
    text = re.sub(r" +", "", text)
    c = Counter(text).most_common(10)
    plt.plot([list(c.keys()), list(c.values())])
    
    

def word_cloud(text, language, mask_path=None, 
               save_path=None,max_font_size=50, 
               max_words=100, background_color="white", 
               interpolation='bilinear', mask_type="color", 
               contour_width=3, contour_color='firebrick'):
  
    if mask_path is not None and os.path.isfile(mask_path) and mask_type == "color":
        mask = np.array(Image.open(mask_path))
        mask[mask == 0] = 255
        wc = WordCloud(max_font_size=max_font_size, 
                       max_words=max_words, 
                       background_color=background_color, 
                       mask=mask).generate(text)
        image_colors = ImageColorGenerator(mask)
        wc = wc.recolor(color_func=image_colors)

    elif mask_path is not None and os.path.isfile(mask_path) and mask_type == "shape":
        mask = np.array(Image.open(mask_path))
        wc = WordCloud(max_font_size=max_font_size, 
                       max_words=max_words, 
                       background_color=background_color, 
                       mask=mask,
                       contour_width=contour_width, 
                       contour_color=contour_color).generate(text)
    else:
        wc = WordCloud(max_font_size=max_font_size, 
                       max_words=max_words, 
                       background_color=background_color).generate(text)
    plt.figure(figsize=(25, 25))
    plt.imshow(wc, interpolation=interpolation)
    plt.axis("off")
    plt.title("Word Cloud for %s." % language)
    plt.savefig(save_path, format="png")
    # plt.show()
    plt.close()

def show_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(32, 32))
    ax = sns.heatmap(cm, annot = True, fmt = "d")

    ax.set_xlabel('Predicted Language')
    ax.set_ylabel('Actual Language')
    ax.set_title('Language Identification Confusion Matrix', fontweight='bold', fontsize=15, pad=10)
    ax.set_xticklabels(languages, fontweight='bold')# rotation=90
    ax.set_yticklabels(languages, fontweight='bold')
    plt.show()