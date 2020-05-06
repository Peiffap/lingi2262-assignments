#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from collections import Counter

from pandas import read_csv

from torch import nn
import torch.nn.functional as F

# Classifiers.
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from skorch import NeuralNetClassifier

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV

from sklearn.metrics import balanced_accuracy_score, make_scorer

from sklearn.preprocessing import LabelEncoder, scale, normalize

from sklearn.impute import SimpleImputer, KNNImputer

train_file = "../data/ML-A5-2020_train.csv"
test_file = "../data/ML-A5-2020_test.csv"

seed = 0

# Read the data into a pandas dataframe.
train_df = read_csv(train_file, index_col=0)

# Encore categorical data (one-hot).
to_modify = ['patient', 'tissue', 'level.mito', 'level.ribo', 'low.yield', 'marker.A', 'marker.B', 'marker.C','marker.D', 'marker.E', 'marker.F', 'marker.G']
enc = LabelEncoder()
for i in to_modify:
    train_df[i] = enc.fit_transform(train_df[i])

# PyTorch requires this.
train_df['label'] = [(i + 1) // 2 for i in train_df['label']]

# Convert the dataframe to a NumPy array.
# and split into inputs and outputs.
train_np = train_df.values
X = train_np[:, 1:-1].astype(np.float32)
y = train_np[:, -1].astype(np.int64)

# Fill missing values.
imputer = SimpleImputer()
X = imputer.fit_transform(X)

normalized_X = normalize(X)
standardized_X = scale(X)

# Define the model evaluation procedure.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

# Evaluate the model and report the mean performance.

class MyModule(nn.Module):
    def __init__(self, num_units=len(X[0]), nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(num_units, 100)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(100, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.relu(self.dense1(X))
        X = F.softmax(self.output(X), dim=-1)
        return X


net = NeuralNetClassifier(
    MyModule,
    max_epochs=10,
    lr=0.1,
    # Shuffle training data on each epoch
    iterator_train__shuffle=True
)

model_names = [#'DL',
               #'bagging',
               #'MLP',
               #'adaboost',
               #'QDA',
               #'LDA',
               #'random forest',
               #'SVM rbf',
               #'SVM linear',
               #'SVM sigmoid',
               #'decision tree',
               #'KNN',
               'xgboost']

ctr = Counter(y)
ratio = ctr[0] / ctr[1]

models = [#net,
          #BaggingClassifier(random_state=seed),
          #MLPClassifier(random_state=seed),
          #AdaBoostClassifier(random_state=seed),
          #QuadraticDiscriminantAnalysis(),
          #LinearDiscriminantAnalysis(),
          #RandomForestClassifier(random_state=seed),
          #SVC(kernel='rbf'),
          #SVC(kernel='linear'),
          #SVC(kernel='sigmoid'),
          #DecisionTreeClassifier(random_state=seed),
          #KNeighborsClassifier(),
          XGBClassifier(seed=seed, scale_pos_weight=ratio, max_depth=2, n_estimators=140, learning_rate=0.01)]

bcr = 'balanced_accuracy'

for model, name in zip(models, model_names):
    # result_acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    # print('Accuracy: %.3f' % result_acc.mean())

    print('BCR for %s:' % name)

    bcr_raw = cross_val_score(model, X, y, cv=cv, scoring=bcr)
    print('Raw:          %.3f' % bcr_raw.mean())

    bcr_normalized = cross_val_score(model, normalized_X, y, cv=cv, scoring=bcr)
    print('Normalized:   %.3f' % bcr_normalized.mean())

    bcr_standardized = cross_val_score(model, standardized_X, y, cv=cv, scoring=bcr)
    print('Standardized: %.3f' % bcr_standardized.mean())

    print()