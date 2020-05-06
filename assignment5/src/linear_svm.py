#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from pandas import read_csv

from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper, gen_features

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import pickle
import matplotlib.pyplot as plt

plt.style.use("ggplot")

train_file = "../data/ML-A5-2020_train.csv"
test_file = "../data/ML-A5-2020_test.csv"

seed = 0

print('Reading data')

# Read the data into a pandas dataframe.
train_df = read_csv(train_file, index_col=0)

X = train_df.copy()
y = X.pop('label')

print('Encoding categorical features')

categorical_features = ['patient', 'tissue']
boolean_features = ['low.yield', 'marker.A', 'marker.B', 'marker.C','marker.D', 'marker.E', 'marker.F', 'marker.G']
ordinal_features = ['level.mito', 'level.ribo']

numeric_features = X.columns.values.tolist()

for i in boolean_features:
    X[i] = X[i].astype(int)

for i in categorical_features + boolean_features + ordinal_features:
    numeric_features.remove(i)

# Uses extracts from https://github.com/kinir/catboost-with-pipelines/blob/master/sklearn-pandas-catboost.ipynb.

gen_category = gen_features(
    columns=[[i] for i in categorical_features + boolean_features],
    classes=[
        {
            "class": SimpleImputer,
            "strategy": "most_frequent"
        },
        {
            "class": OneHotEncoder
        }
    ]
)

gen_category_ord_enc = gen_features(
    columns=[[i] for i in categorical_features + boolean_features],
    classes=[
        {
            "class": SimpleImputer,
            "strategy": "most_frequent"
        },
        {
            "class": OrdinalEncoder,
            "dtype": np.int8
        }
    ]
)

gen_ordinal = gen_features(
    columns=[[i] for i in ordinal_features],
    classes=[
        {
            "class": SimpleImputer,
            "strategy": "most_frequent"
        },
        {
            "class": OrdinalEncoder,
            "dtype": np.int8
        }
    ]
)

gen_numeric = gen_features(
    columns=[[i] for i in numeric_features],
    classes=[
        {
            "class": SimpleImputer,
            "strategy": "most_frequent"
        }
    ]
)

preprocess_mapper1 = DataFrameMapper(
    [
        *gen_category,
        *gen_ordinal,
        *gen_numeric
    ],
    input_df=True,
    df_out=True
)

preprocess_mapper2 = DataFrameMapper(
    [
        *gen_category_ord_enc,
        *gen_ordinal,
        *gen_numeric
    ],
    input_df=True,
    df_out=True
)

print('Building models')

class CustomCatBoostClassifier(CatBoostClassifier):

    def fit(self, X, y=None, **fit_params):
        return super().fit(
            X,
            y=y,
            **fit_params
        )

class CustomFeatureSelection(SelectFromModel):

    def transform(self, X):
        # Get indices of important features
        important_features_indices = list(self.get_support(indices=True))

        # Select important features
        _X = X.iloc[:, important_features_indices].copy()

        return _X

xgb = SelectFromModel(XGBClassifier(n_estimators=100,
                                    booster="gblinear",
                                    scale_pos_weight=885/115,
                                    max_depth=4,
                                    verbosity=3),
                      max_features=60)

cb = CustomFeatureSelection(CustomCatBoostClassifier(n_estimators=100,
                                                     class_weights=[1, 885/115],
                                                     max_depth=4,
                                                     cat_features=categorical_features + ordinal_features + boolean_features),
                            max_features=60)

feat_select = FeatureUnion([("pca", make_pipeline(preprocess_mapper1,
                                                  PCA(n_components=60, whiten=True))),
                            ("xgb", make_pipeline(preprocess_mapper1,
                                                  xgb)),
                            ("cb", make_pipeline(preprocess_mapper2,
                                                 cb)),
                            ("kbest", make_pipeline(preprocess_mapper1,
                                                    VarianceThreshold(),
                                                    SelectKBest(chi2, k=60)))
                            ])

svm = LinearSVC(class_weight='balanced',
                C=0.1,
                max_iter=10000,
                dual=False)

estimator = BaggingClassifier(base_estimator=svm, n_estimators=150)

model = make_pipeline(feat_select,
                      StandardScaler(),
                      PCA(n_components=30, whiten=True),
                      estimator)

print('Preparing cross-validation')

# Evaluate the model and report the mean performance.

print('Evaluating models')

folds = list(range(3, 11))

res = {}

for i in folds:
    # Define the model evaluation procedure.
    cv = RepeatedStratifiedKFold(n_splits=i, n_repeats=1, random_state=seed)
    print("Number of folds: %d" % i)

    scores = cross_validate(model, X, y, cv=cv, scoring='balanced_accuracy',
                            return_train_score=True, error_score='raise')

    print('Test : %.3f (+/- %.3f)' % (scores['test_score'].mean(), scores['test_score'].std() * 2))
    print('Train: %.3f (+/- %.3f)' % (scores['train_score'].mean(), scores['train_score'].std() * 2))

    res[i] = {}

    res[i]['test'] = scores['test_score']
    res[i]['train'] = scores['train_score']


pickle.dump(res, open("../report/data/res.p", "wb"))

