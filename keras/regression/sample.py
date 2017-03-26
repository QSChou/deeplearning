
# reference http://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# load dataset
#dataframe = pandas.read_csv("~/Documens/keras/data/housing.csv",delim_whitespace=True, header=None)
dataframe = pandas.read_csv("housing.csv",delim_whitespace=True, header=None)

dataset = dataframe.values
#Split into input(x) and output(y) variables
x = dataset[:,0:13]
y = dataset[:,13]

# define base mode
def baseline_model():
#create model
 model=Sequential()
 model.add(Dense(13, input_dim=13, init='normal', activation='relu'))
 model.add(Dense(1, init='normal'))
#Compile model
 model.compile(loss='mean_squared_error', optimizer='adam')
 return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x, y, cv=kfold)

print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
