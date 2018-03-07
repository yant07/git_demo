# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:00:31 2018

@author: YANT07
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
#from keras.callbacks import EarlyStopping

import pandas as pd
import os
os.getcwd()
os.chdir("C:/Users/yant07/Desktop/NLP")
df_train=pd.read_csv("training1.csv")
df_test=pd.read_csv("test1.csv")

x_train=df_train.iloc[:,3: ]
x_test=df_test.iloc[:,3: ]
y_train=df_train.iloc[:,1:2]
y_test=df_test.iloc[:,1:2]

y_train_dummy=pd.get_dummies(y_train)
y_test_dummy=pd.get_dummies(y_test)

# Generate dummy data
import numpy as np
#x_train = np.random.random((1000, 20))
#y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
#x_test = np.random.random((100, 20))
#y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=7148))
model.add(Dropout(0.6))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(5, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#early_stopping=EarlyStopping(monitor="val_loss",patience=2)

model.fit(x_train, y_train_dummy,
          epochs=30,
          batch_size=100)
#          ,callbacks=[early_stopping])
score = model.evaluate(x_test, y_test_dummy, batch_size=100)
          
from keras.models import load_model
model.save("test.h5")
model1=load_model("test.h5")
score1 = model1.evaluate(x_test, y_test_dummy, batch_size=100)


model.fit(x_train, y_train_dummy,
          epochs=20,
          batch_size=80,
          validation_split=0.2)
score = model.evaluate(x_test, y_test_dummy, batch_size=100)

pred=model.predict_classes(x_test,batch_size=100,verbose=1)
pred1=pd.DataFrame(pred)
from sklearn.metrics import confusion_matrix
tb=confusion_matrix(y_test,pred1)