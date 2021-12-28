# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 21:12:49 2020

@author: GL62M
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 20:33:42 2020

@author: GL62M
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:58:43 2019

@author: GL62M
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Set random seed to make results reproducable
np.random.seed(42)
plt.style.use('seaborn')

data = pd.read_csv('Temp_and_rain.csv')
data.fillna(0,inplace=True)

data= data.sample(100)

import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
#from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.metrics import mean_squared_error 

'''
X=data.drop('CLASS',axis=1)
# One hot encoding
X=pd.get_dummies(X, drop_first=True)
y=data['CLASS']
y=pd.get_dummies(y, drop_first=True)
'''

X = data.drop('rain',axis=1)
y = data[['rain']]


# Train/test split
#train_X, test_X, train_y, test_y = train_test_split(X, y)

lm = LinearRegression()
modellm = lm.fit(X,y)

predictionslm = lm.predict(X)

accuracylm = lm.score(X,y)
print('linear regression score')
print(accuracylm*100,'%')
########### Linear Regression End ######################

########### Logistic Regression Begins #################
########## Adaboost Begin ###############################
am = AdaBoostRegressor()
modelam = am.fit(X,y)

predictionsam = am.predict(X)

accuracyam = am.score(X,y)
print('adaboost regression score')
print(accuracyam*100,'%')
########## Adaboost End ###############################

########## gradient boosting Begin #####################
gm = GradientBoostingRegressor()
modelgm = gm.fit(X,y)

predictionsgm = gm.predict(X)

accuracygm = gm.score(X,y)
print('grad boost regression score')
print(accuracygm*100,'%')

########## gradient boosting end #####################

########## xgboosting Begin #####################
xgm = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.7,
                max_depth = 3, alpha = 10, n_estimators = 15)
modelxgm = xgm.fit(X,y)

predictionsxgm = xgm.predict(X)

accuracyxgm = xgm.score(X,y)
print('x grad boost regression score')
print(accuracyxgm*100,'%')
########## xgboosting end #####################


########## regression tree Begin #####################
rm = DecisionTreeRegressor(criterion='mse',     # Initialize and fit regressor
                             max_depth=3)  
modelrm = rm.fit(X,y)

predictionsrm = rm.predict(X)

accuracyrm = rm.score(X,y)
print('regression tree score')
print(accuracyrm*100,'%')

########## regression tree end #####################

#########Support Vector Regressor Begin################
from sklearn.svm import SVR
regressor=SVR()

regressor.fit(X,y)

predictionSVR = regressor.predict(X);

accuracySVR = regressor.score(X,y)
print('Support Vector Regression Score')
print(accuracySVR*100,'%')

########Support Vector Regressor End###############

##########Random Forest Regressor Begin###########
from sklearn.ensemble import RandomForestRegressor
RF = RandomForestRegressor(n_estimators = 15, criterion='mse', max_depth=3, random_state=1)

RF.fit(X,y)

predictionRF = RF.predict(X)

accuracyRF = RF.score(X, y)
print('Random Forest Regressor Score')
print(accuracyRF*100, '%')

##########Random Forest Regressor End###########
y=y.astype('int')
########### Logistic Regression Begins ######################
logm = LogisticRegression()
modellogm = logm.fit(X,y)

print('=================================================================')
print('Predicted values for target')
predictionslogm = logm.predict(X)
print(predictionslogm)

#print(lm.coef_)
#print(lm.intercept_)

accuracylogm = logm.score(X,y)
print('logistic regression score')
print(accuracylogm*100,'%')

#########Multi Layer Perceptron Regression Begin##########
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(activation='tanh', max_iter=200, solver='adam', learning_rate='adaptive', alpha=1e-5,hidden_layer_sizes=(30, 30, 30 ), random_state=42)

mlp.fit(X, y)
y_pred_naive = mlp.predict(X)
print('=================================================================')
print('Predicted values for target')
print(y_pred_naive)

accuracy_mlp = mlp.score(X, y)
print('Multi Layer Perceptron Score')
print(accuracy_mlp*100, '%')
print('=================================================================')
###########Multi Layer Perceptron Regression End########