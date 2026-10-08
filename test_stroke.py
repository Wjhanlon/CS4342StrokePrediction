import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from stroke_project import train, predict

df = pd.read_csv("diabetes_data.csv")

DROP = ["Stroke", "GenHlth", "DiffWalk", "PhysHlth", "MentHlth"]
FEATURES = [c for c in df.columns if c not in DROP] # all 17 other columns
X = df[FEATURES].values.astype(float)
y = df["Stroke"].values.astype(int)                   # capital S in this file

# split FIRST (shuffled and stratified, since the file is sorted by source)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)

# scale using TRAIN statistics only
mean = Xtrain.mean(axis=0)
std = Xtrain.std(axis=0)
Xtrain = (Xtrain - mean) / std
Xtest = (Xtest - mean) / std

from sklearn.metrics import roc_auc_score

w, b = train(Xtrain, ytrain, alpha=0.0003, n_epoch=10)

Ytr, Ptr = predict(Xtrain, w, b)
Yte, Pte = predict(Xtest, w, b)

print("train accuracy:", (Ytr == ytrain).mean().round(3))
print("test accuracy: ", (Yte == ytest).mean().round(3))
print("test AUC:", round(roc_auc_score(ytest, Pte[:, 1]), 3))


from sklearn.metrics import classification_report

# default cutoff (0.5)
print("--- threshold 0.5 ---")
print(classification_report(ytest, Yte, zero_division=0))

# lower cutoff: call it a stroke if P(stroke) > 0.06
Y_low = (Pte[:, 1] > 0.06).astype(int)
print("--- threshold 0.06 ---")
print(classification_report(ytest, Y_low, zero_division=0))