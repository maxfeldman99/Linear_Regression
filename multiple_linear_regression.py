#Multiple Linear Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values  # this is the matrix of in-dependent variables
y = dataset.iloc[:, 4].values  # this is the dependant variable

# Encoding categorical data
# Encoding the Independent Variable
# The encoding must be done before the splitting
# This section will actually encode our "State" column into number like data since it's a string (dummy variables)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]  # removed the 1st column from x , by putting 1 we starting from index 1 till the end

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# Feature Scaling (no need for multiple regression)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1) # creating new column tor dummy x0 and moving the old matrix inside the new matrix
X_opt = X[:, [0, 3]] # 2,1,4,5 were removed since we found that it has the highest P-value and more then 0.05
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary() # we check who has the highest P-value to remove it later
print(regressor_OLS.summary())





