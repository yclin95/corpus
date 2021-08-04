#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 22:20:22 2021

@author: nick
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import os
import re
import json




df3=pd.read_csv(r'out.csv')

from nltk.sentiment.vader import SentimentIntensityAnalyzer

import pandas as pd
vader = SentimentIntensityAnalyzer()

# Iterate through the headlines and get the polarity scores
scores = [vader.polarity_scores(comments) for comments in df3.clean_comments.values]
# Convert the list of dicts into a DataFrame
scores_df = pd.DataFrame(scores)
# Join the DataFrames

scores_df.reset_index(drop=True, inplace=True)
df3.reset_index(drop=True, inplace=True)
df = pd.concat([df3, scores_df['compound']], axis=1)

scored_news = pd.concat([df3, scores_df['compound']], axis=1)

#compound=0.05
scored_news1=scored_news.iloc[:,[4,16]]

scored_news1['compound']= np.int64(scored_news1['compound']<0.05)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(scored_news1['accepted'], scored_news1['compound'])
print(cm2)

from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from scikitplot.estimators import plot_feature_importances
from scikitplot.metrics import plot_confusion_matrix, plot_roc
print(confusion_matrix(scored_news1['accepted'], scored_news1['compound']))
skplt.metrics.plot_confusion_matrix(scored_news1['accepted'], scored_news1['compound'],figsize=(8,8))


from sklearn.metrics import accuracy_score
p=accuracy_score(scored_news1['accepted'], scored_news1['compound'])
from sklearn.metrics import f1_score
f1=f1_score(scored_news1['accepted'], scored_news1['compound'])
from sklearn.metrics import recall_score
recall=recall_score(scored_news1['accepted'], scored_news1['compound'])
print(p)
print(f1)
print(recall)
#导入要用的库
import sklearn.metrics as metrics 
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score as AUC
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(scored_news1['accepted'], scored_news1['compound'])
roc_auc=metrics.auc(fpr,tpr)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('ROC curve of lexicon-base method')
plt.legend(loc="lower right")

C=confusion_matrix(scored_news1['accepted'], scored_news1['compound'])
df=pd.DataFrame(C,index=["ant", "bird", "cat"],columns=["ant", "bird", "cat"])
sns.heatmap(df,annot=True)



#

#compound=0.05
scored_news1=scored_news.iloc[:,[4,16]]

scored_news1['compound']= np.int64(scored_news1['compound']<-0.05)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(scored_news1['accepted'], scored_news1['compound'])
print(cm2)

from sklearn.metrics import confusion_matrix
import scikitplot as skplt
from scikitplot.estimators import plot_feature_importances
from scikitplot.metrics import plot_confusion_matrix, plot_roc
print(confusion_matrix(scored_news1['accepted'], scored_news1['compound']))
skplt.metrics.plot_confusion_matrix(scored_news1['accepted'], scored_news1['compound'],figsize=(8,8))


from sklearn.metrics import accuracy_score
p=accuracy_score(scored_news1['accepted'], scored_news1['compound'])
from sklearn.metrics import f1_score
f1=f1_score(scored_news1['accepted'], scored_news1['compound'])
from sklearn.metrics import recall_score
recall=recall_score(scored_news1['accepted'], scored_news1['compound'])
print(p)
print(f1)
print(recall)
#导入要用的库

import sklearn.metrics as metrics 
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score as AUC
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(scored_news1['accepted'], scored_news1['compound'])
roc_auc=metrics.auc(fpr,tpr)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Plot ROC curve
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.title('ROC curve of lexicon-base method(-0.05)')
plt.legend(loc="lower right")

C=confusion_matrix(scored_news1['accepted'], scored_news1['compound'])
df=pd.DataFrame(C,index=["ant", "bird", "cat"],columns=["ant", "bird", "cat"])
sns.heatmap(df,annot=True)



