#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
import random as rn
import os
import numpy as np
from pandas import read_csv
from keras.layers import Dense
from keras.models import Sequential
import pandas as pd


# In[2]:


os.environ['PYTHONHASHEED'] = '0'
np.random.seed(37)
rn.seed(1254)
tf.random.set_seed(89)


# In[3]:


# load the data set
filename = 'pima-indians-diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values


# In[4]:


#split data into x and y variables
x = array[:, 0:8]
y = array[:, 8]


# In[5]:


# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation = 'sigmoid'))


# In[6]:


## compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[7]:


## fit the keras model on the dataset
history = model.fit(x, y, epochs=150, validation_split=0.33, batch_size=10)


# In[8]:


# evaluate the keras model
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[9]:


# list all data is history
print(history.history.keys())


# In[10]:


## summerize the history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[11]:


## summerize the history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# #### hyper tunning the forest model

# In[11]:


data2 = pd.read_csv('forestfires.csv')


# In[12]:


data2


# In[18]:


data2.drop(columns=['month','day'], axis=1, inplace=True)


# In[19]:


data2


# In[20]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
data2["size_category"] = label_encoder.fit_transform(data2["size_category"])
data2


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(data2)
plt.show()


# In[21]:


from sklearn.preprocessing import LabelEncoder


# In[22]:


x = data2.iloc[:, :-1]
x


# In[23]:


y = data2.iloc[:, -1]
y


# In[24]:


import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


# In[25]:


## standardized the data
a = StandardScaler()
a.fit(x)
x_stand = a.transform(x)


# In[26]:


x = pd.DataFrame(x)


# In[27]:


x


# In[28]:


## create model
model = Sequential()
model.add(Dense(20, input_dim=28, activation='relu'))   ## 1st layer taking activation function is randomly not doing hypertinning
model.add(Dense(10,  activation='relu')) ## 2nd layer ## randomly putting neurons
model.add(Dense(10,  activation='relu'))
model.add(Dense(1,  activation='sigmoid'))


# In[29]:


## compile model - estabilish connection between input and output layer
model.compile(loss='binary_crossentropy',  optimizer='adam', metrics=['accuracy'])   ## binary beacuse there are only two classes in dependent column


# In[30]:


## fit the model
history = model.fit(x, y, validation_split=0.33, epochs=250, batch_size=10)


# In[31]:


## evaluate the model
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[32]:


## summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[33]:


## summerize history for loss
plt.plot(history.history['loss'])              ## while training and testing loss
plt.plot(history.history['val_accuracy'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ## Now data is ready for the hyper tunning

# In[5]:


from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot


# In[6]:


from sklearn.datasets import make_blobs 


# In[27]:


data2


# In[7]:


# create NN and find best batch size and best epchs
# Importing the necessary packages
from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential # In NN we are going to add components (i.e. hidden layers) one by one in sequential manner
from keras.layers import Dense # the no. of neurons
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
#from keras.optimizers import adam_v2
#from tensorflow.keras.optimizers import Adam # Adam - Adaptive Momentum - is an optimizer


# In[8]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[28]:


from keras.layers import Dropout
# drop out rate - to control the accuracy.to overcome overfit of model randomly remove connecction of some neurons 

# define the model foe learning rate and dropout rate

def create_model(learning_rate, dropout_rate):
    model = Sequential()
    model.add(Dense(8, input_dim = 28, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(6, input_dim = 28, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dropout(dropout_rate))
   
    adam = Adam(learning_rate = learning_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model


# In[31]:


model = KerasClassifier(build_fn = create_model, verbose = 0, batch_size = 10, epochs = 150)


# In[33]:


learning_rate = [0.0001,0.00001]
dropout_rate = [0.0, 0.1]  ## drop 0% , 10%, 20% neurons


# In[36]:


param_grid = dict(learning_rate = learning_rate, dropout_rate = dropout_rate )


# In[37]:


grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(), verbose = 10)


# In[38]:


grid_result = grid.fit(x,y)        # total (3*3)*5 default folds = 45 , 


# In[39]:


# Summarize the results
print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param))


# #### Tuning of Hyperparameters:- Activation Function and Kernel Initializer

# In[40]:


# Defining the model

def create_model(activation_function,init):
    model = Sequential()
    model.add(Dense(8,input_dim = 28,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(0.0))
    model.add(Dense(12,input_dim = 28,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(0.0))
    model.add(Dense(1,activation = 'sigmoid'))

    adam = Adam(learning_rate = 0.5)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model
    # In machine learning, Loss function is used to find error or deviation in the learning process.
    # Keras requires loss function during model compilation process. https://www.tutorialspoint.com/keras/keras_model_compilation.htm

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 50)

# Define the grid search parameters
activation_function = ['softmax','relu','tanh','linear'] # find which activation function is best out of these
init = ['uniform','normal','zero'] # Weight initializers from where the weights has to be sampled randomly. Uniform distribution, normal distribution and zero distribution

# Make a dictionary of the grid search parameters
param_grids = dict(activation_function = activation_function,init = init)

# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,param_grid = param_grids,cv = KFold(),verbose = 10)
grid_result = grid.fit(x,y)



# In[41]:


# Summarize the results
print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param))


# #### Tunning no of hidden layers

# In[42]:


# Defining the model

def create_model(neuron1,neuron2):
    model = Sequential()
    model.add(Dense(neuron1,input_dim = 28,kernel_initializer = 'uniform',activation = 'linear'))
    model.add(Dropout(0.0))
    model.add(Dense(neuron2,input_dim = neuron1,kernel_initializer = 'uniform',activation = 'linear'))
    model.add(Dropout(0.0))
    model.add(Dense(1,activation = 'sigmoid'))

    adam = Adam(learning_rate = 0.001)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 50)

# Define the grid search parameters

neuron1 = [4,8,16]# in first hidden layer use 4,8,16 neurons
neuron2 = [2,4,8]# in second hidden layer use 2,4,8 neurons

# Make a dictionary of the grid search parameters

param_grids = dict(neuron1 = neuron1,neuron2 = neuron2)

# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,param_grid = param_grids,cv = KFold(),verbose = 10)
grid_result = grid.fit(x,y)


# In[43]:


# Summarize the results
print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param))


# # #### Training model with optimum values of Hyperparameters

# In[44]:


#skip this
from sklearn.metrics import classification_report, accuracy_score

# Defining the model

def create_model():
    model = Sequential()
    model.add(Dense(4,input_dim = 28,kernel_initializer = 'uniform',activation = 'linear'))
    model.add(Dropout(0.0))
    model.add(Dense(2,input_dim = 28,kernel_initializer = 'uniform',activation = 'linear'))
    model.add(Dropout(0.0))
    model.add(Dense(1,activation = 'sigmoid'))

    adam = Adam(learning_rate = 0.5) #sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 10,epochs = 150)

# Fitting the model

model.fit(x,y)

# Predicting using trained model

y_predict = model.predict(x)

# Printing the metrics
print(accuracy_score(y,y_predict))


# In[45]:


# Defining the model

def create_model(activation_function,init):
    model = Sequential()
    model.add(Dense(8,input_dim = 28,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(0.0))
    model.add(Dense(12,input_dim = 28,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(0.0))
    model.add(Dense(1,activation = 'sigmoid'))

    adam = Adam(learning_rate = 0.5)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model
    # In machine learning, Loss function is used to find error or deviation in the learning process.
    # Keras requires loss function during model compilation process. https://www.tutorialspoint.com/keras/keras_model_compilation.htm

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 10,epochs = 150)

# Define the grid search parameters
activation_function = ['softmax','relu','tanh','linear'] # find which activation function is best out of these
init = ['uniform','normal','zero'] # Weight initializers from where the weights has to be sampled randomly. Uniform distribution, normal distribution and zero distribution

# Make a dictionary of the grid search parameters
param_grids = dict(activation_function = activation_function,init = init)

# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,param_grid = param_grids,cv = KFold(),verbose = 10)
grid_result = grid.fit(x,y)



# In[ ]:





# # Hyperparameters all at once

# 
# The hyperparameter optimization was carried out by taking 2 hyperparameters at once. We may have missed the best values. The performance can be further improved by finding the optimum values of hyperparameters all at once given by the code snippet below.
# #### This process is computationally expensive.

# In[ ]:





# In[ ]:





# In[ ]:


def create_model(learning_rate,dropout_rate,activation_function,init,neuron1,neuron2):
    model = Sequential()
    model.add(Dense(neuron1,input_dim = 28,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neuron2,input_dim = neuron1,kernel_initializer = init,activation = activation_function))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation = 'sigmoid'))

    adam = Adam(learning_rate = learning_rate)
    model.compile(loss = 'binary_crossentropy',optimizer = adam,metrics = ['accuracy'])
    return model

# Create the model

model = KerasClassifier(build_fn = create_model,verbose = 0)

# Define the grid search parameters

batch_size = [10,20,40]
epochs = [10,50,150]
learning_rate = [0.001,0.01,0.1]
dropout_rate = [0.0,0.1,0.2]
activation_function = ['softmax','relu','tanh','linear']
init = ['uniform','normal','zero']
neuron1 = [4,8,16]
neuron2 = [2,4,8]

# Make a dictionary of the grid search parameters

param_grids = dict(batch_size = batch_size,epochs = epochs,learning_rate = learning_rate,dropout_rate = dropout_rate,
                   activation_function = activation_function,init = init,neuron1 = neuron1,neuron2 = neuron2)

# Build and fit the GridSearchCV

grid = GridSearchCV(estimator = model,param_grid = param_grids,cv = KFold(),verbose = 10)
grid_result = grid.fit(x,y)

# Summarize the results
print('Best : {}, using {}'.format(grid_result.best_score_,grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
  print('{},{} with: {}'.format(mean, stdev, param))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




