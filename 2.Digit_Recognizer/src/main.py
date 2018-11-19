# coding=utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv('../data/train.csv')

Y_train_series = train_df['label']
X_train_df = train_df.drop(labels='label', axis=1)

del train_df

X_train_df = X_train_df / 255.0

X_train_ndarray = X_train_df.values.reshape(-1, 28, 28, 1)

print(X_train_ndarray.shape)

plt.imshow(X_train_ndarray[5][:, :, 0])
plt.show()
