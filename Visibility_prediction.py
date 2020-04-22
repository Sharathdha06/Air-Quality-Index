import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('Real_Combine.csv')
df.head()
df.describe()
df.info()

df.isnull().sum()
df['PM 2.5'].isnull().sum()
df['PM 2.5'] = df['PM 2.5'].fillna(df['PM 2.5'].mean())

sns.pairplot(df)
sns.scatterplot(df['T'],df['PM 2.5'])
sns.scatterplot(df['TM'],df['PM 2.5'])
sns.scatterplot(df['Tm'],df['PM 2.5'])
sns.scatterplot(df['SLP'],df['PM 2.5'])
sns.scatterplot(df['H'],df['PM 2.5'])
sns.scatterplot(df['VV'],df['PM 2.5'])
sns.scatterplot(df['V'],df['PM 2.5'])
sns.scatterplot(df['VM'],df['PM 2.5'])

#Checking correlation of data
corr = df.corr()
plt.figure(figsize = (20,20))
heat_corr = sns.heatmap(corr, annot = True, cmap = 'RdYlGn')

#Splitting data
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
#Feature importance

from sklearn.ensemble import ExtraTreesRegressor
ETR = ExtraTreesRegressor()
model = ETR.fit(X,y)

print(model.feature_importances_)
Feature_imp = pd.Series(model.feature_importances_, index = X.columns)
Feature_imp.nlargest(10).plot(kind = 'barh')

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 66)
#Model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 400, max_features = 'auto', max_depth = 10)
regressor.fit(x_train,y_train)

train_R_squared = regressor.score(x_train,y_train)
test_R_squared = regressor.score(x_test,y_test)

from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor,x_train,y_train,cv = 10)
score.mean()

prediction = regressor.predict(x_test)

from sklearn import metrics
MAE = metrics.mean_absolute_error(y_test,prediction)
MSE = metrics.mean_squared_error(y_test,prediction)
RMSE = np.sqrt(MSE)

print('Mean Absolute error is :',MAE)
print('Mean Square error is :',MSE)
print('Root Mean Square error is :',RMSE)

#Hyper para meter tuning
# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':[10,50,100,200,300,400,500,600,700,800,900,1000],'max_features':['auto','sqrt'],
              'max_depth':[1,2,3,4,5,6,7,8,9,10]}
grid_search = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_




#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import ReLU, LeakyReLU

NN_model = Sequential()
#Input layer
NN_model.add(Dense(128, kernel_initializer = 'glorot_uniform',activation ='relu',input_dim = x_train.shape[1] ))

#Hidden Layers
NN_model.add(Dense(256,activation = 'relu'))
NN_model.add(Dense(256,activation = 'relu'))
NN_model.add(Dense(256,activation = 'relu'))
NN_model.add(Dense(256,activation = 'relu'))

#Output Layer
NN_model.add(Dense(1,activation = 'linear'))

#Compile NN layers
NN_model.compile(loss = 'mean_absolute_error', optimizer = 'adam',metrics = ['mean_absolute_error'])

NN_model.fit(x_train, y_train,validation_split = 0.2, batch_size = 25, epochs = 100)

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    NN_model = Sequential()
    NN_model.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    NN_model.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    NN_model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    NN_model.compile(optimizer = 'adam', loss = 'mean_absolute_error', metrics = ['accuracy'])
    return NN_model
NN_model = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
Mean_absolute_errors = cross_val_score(estimator = NN_model, X = x_train, y = y_train, cv = 10, n_jobs = -1)
mean = Mean_absolute_errors.mean()
variance = Mean_absolute_errors.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    NN_model = Sequential()
    NN_model.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    NN_model.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    NN_model.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    NN_model.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu'))
    NN_model.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))
    NN_model.compile(optimizer = optimizer, loss = 'mean_absolute_error', metrics = ['mean_absolute_error'])
    return NN_model
NN_model = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = NN_model,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(x_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

