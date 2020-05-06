#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Deep learning approach.

from collections import Counter

import numpy as np
from scipy import sparse

from pandas import read_csv

from torch import nn, sigmoid, FloatTensor
import torch.nn.functional as F

from skorch import NeuralNetClassifier, NeuralNetBinaryClassifier
from skorch.callbacks import EpochScoring

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, GridSearchCV

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import balanced_accuracy_score, make_scorer

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.decomposition import PCA

from sklearn.pipeline import make_pipeline, FeatureUnion

from sklearn.impute import SimpleImputer, KNNImputer

from xgboost import XGBClassifier

train_file = "../data/ML-A5-2020_train.csv"
test_file = "../data/ML-A5-2020_test.csv"

seed = 0

# Read the data into a pandas dataframe.
train_df = read_csv(train_file, index_col=0)

# Encore categorical data.
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

xgb = SelectFromModel(XGBClassifier(n_estimators=100,
                                    scale_pos_weight=7.7,
                                    booster="gblinear"),
                      max_features=150)


pca = PCA(n_components=150, whiten=True)

# Define the model evaluation procedure.
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

# Evaluate the model and report the mean performance.

class MyModule(nn.Module):
    def __init__(self, sizes=[300, 60, 2], nonlin=F.relu):
        super(MyModule, self).__init__()

        self.dense0 = nn.Linear(sizes[0], sizes[1])
        self.nonlin = nonlin
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(sizes[-2], sizes[-1])

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        X = self.dropout(X)
        X = F.softmax(self.output(X), dim=-1)
        return X

net = NeuralNetClassifier(
    MyModule,
    max_epochs=100,
    lr=0.008,
    criterion__weight=FloatTensor([0.115, 0.885]),
    # Shuffle training data on each epoch
    iterator_train__shuffle=True
)

model = make_pipeline(SimpleImputer(strategy="most_frequent"),
                      StandardScaler(),
                      FeatureUnion([("xgb", xgb), ("pca", pca)]),
                      net)


scores = cross_validate(model, X, y, cv=cv, scoring='balanced_accuracy', verbose=1, return_train_score=True)
print('Test  : %.3f (+/- %.3f)' % (scores['test_score'].mean(), scores['test_score'].std() * 2))
print('Train: %.3f (+/- %.3f)' % (scores['train_score'].mean(), scores['train_score'].std() * 2))