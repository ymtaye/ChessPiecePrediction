import csv
import pandas as pd
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import os
data = pd.read_csv('final2.csv')
import matplotlib.pyplot as plt
import numpy as np

ADA = []
RF = []
SVM = []
MLP = []

for index, row in data.iterrows():

    if row[0] == 'ADA':
        ADA.append(float(row[1]))
    elif row[0] == 'RF':
        RF.append(float(row[1]))
    elif row[0] == 'SVM':
        SVM.append(float(row[1]))
    elif row[0] == 'MLP':
        MLP.append(float(row[1]))

Below50= []
Is75 = []
Is85 = []
Is90 = []
Is95 = []
Is97 = []

for index, row in data.iterrows():
    if float(row[6]) == 0.5:
        Below50.append(float(row[1]))
    elif float(row[6]) == 0.85:
        Is85.append(float(row[1]))
    elif float(row[6]) == 0.9:
        Is90.append(float(row[1]))
    elif float(row[6]) == 0.95:
        Is95.append(float(row[1]))
    elif float(row[6]) == 0.97:
        Is97.append(float(row[1]))
'''
up = []
down = []
for index, row in data.iterrows():
    if (row['Sampling']) == 'up':
        up.append(row['f1'])
    elif (row['Sampling']) == 'down':
        down.append(row['f1'])

'''

LabelsC = [5,10]
#Labels = ['ADA', 'RF', 'MLP', 'SVM']
#LabelsS = ['UP-SAMPLING', 'DOWN-SAMPLING']

data = [ADA, RF, MLP, SVM]
data2 = [Below50, Is75, Is85, Is90, Is95, Is97]
Labels = ['<=50%', '50%-75%', '75%-85%', '85%-90%', '90%-95%', '95%-97%']


fig7, ax7 = plt.subplots()
ax7.set_title('Variance Explained vs Accuracy')
blot = ax7.boxplot(data2, labels=Labels, patch_artist=True)
#colors = ['green',  'red']
#colors = ['orange', 'red']
colors = ['green',  'yellow', 'red', 'blue', 'orange', 'purple']

task = 'Variance Explained by components' #Resampling Technique'

for patch, color in zip(blot['boxes'], colors):
    patch.set_facecolor(color)

ax7.yaxis.grid(True)
ax7.set_xlabel(task)
ax7.set_ylabel('Accuracy')

plt.show()




