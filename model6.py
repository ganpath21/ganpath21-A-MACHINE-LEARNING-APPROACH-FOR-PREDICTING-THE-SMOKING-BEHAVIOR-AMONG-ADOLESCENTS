from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

root = os.path.dirname(__file__)
path_df = os.path.join(root, 'recons_dataset/dataset6.csv')
data = pd.read_csv(path_df)

scaler = MinMaxScaler()
train, test = train_test_split(data, test_size=0.10)

X_train = train.drop('Q45', axis=1)
Y_train = train['Q45']

X_test = test.drop('Q45', axis=1)
Y_test = test['Q45']

# We don't scale targets: Y_test, Y_train as SVC returns the class labels not probability values
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

clf1 = GradientBoostingClassifier()

# Training the classifier
clf1.fit(X_train, Y_train)


# Testing model accuracy. Average is taken as test set is very small hence accuracy varies a lot everytime the model is trained
acc = 0
acc_binary = 0
for i in range(0, 200):
    Y_hat = clf1.predict(X_test)
    Y_hat_bin = Y_hat>0
    Y_test_bin = Y_test>0
    acc = acc + accuracy_score(Y_hat, Y_test)
    acc_binary = acc_binary + accuracy_score(Y_hat_bin, Y_test_bin)

print("Average test Accuracy:{}".format(acc/200))
print("Average binary accuracy:{}".format(acc_binary/200))
x = np.array([5,2,2,1,1,1,1,1,1,1,2,1,1,1,2,5,1,4,1,3,3,2,2,2,2,3,2,3,3,1,5,4,3,1,5,3,7,3,4,2,2]).reshape(1, -1)
print(clf1.predict(x))
# Saving the trained model for inference
model_path = os.path.join(root, 'models/rfc6.sav')
joblib.dump(clf1, model_path)

# Saving the scaler object
scaler_path = os.path.join(root, 'models/scaler6.pkl')
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

