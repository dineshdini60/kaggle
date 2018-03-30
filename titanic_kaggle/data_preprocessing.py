# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
datatrain = pd.read_csv('C:/Users/chall/Desktop/edu/ml/themljourneybegins/kaggle/titanic/train.csv')
datatest = pd.read_csv('C:/Users/chall/Desktop/edu/ml/themljourneybegins/kaggle/titanic/test.csv')
y_train = datatrain.iloc[:,[1]].values
y = np.ravel(y_train)

X = datatrain.iloc[:,[2,4,5,6,7,9]].values
X_test_out = datatest.iloc[:,[1,3,4,5,6,8]].values

#dealing with null values
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values= 'NaN', strategy="mean",axis= 0)
X[:,[2]] = imp.fit_transform(X[:,[2]])
X_test_out[:,[2]] = imp.fit_transform(X_test_out[:,[2]])
imp_fare = Imputer(missing_values= 'NaN', strategy="mean",axis= 0)
X_test_out[:,[-1]] = imp.fit_transform(X_test_out[:,[-1]])


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_test_out = LabelEncoder()
X_test_out[:, 1] = labelencoder_X_test_out.fit_transform(X_test_out[:, 1])


#test_train split
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_test_out = sc.transform(X_test_out)


