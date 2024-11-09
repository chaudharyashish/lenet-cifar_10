import keras
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from skimage.transform import resize
from keras.applications import vgg16
from keras.datasets import mnist
import numpy as np
from torchvision import transforms,datasets
import torchvision
import pandas as pd
import visualkeras
import cv2
(trainX, trainy), (testX, testy) = keras.datasets.cifar10.load_data()
xtrain=[]
for i in range(0,len(trainX)):
  xtrain.append(resize(trainX[i], (227,227)))
outputClass=["airplane",
"automobile",
"bird",
"cat",
"deer",
"dog",
"frog",
"horse",
"ship",
"truck"]
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
for i in range(16):
 plt.subplot(4,4,i+1)
 plt.title(outputClass[trainy[i][0]])
 plt.imshow(trainX[i])
 plt.axis("off")
plt.show()
trainy = to_categorical(trainy)
testy = to_categorical(testy)
model=Sequential()
model.add(Conv2D(filters=8,kernel_size=(5,5),activation='relu',input_shape=(32,32,3)))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(filters=16,kernel_size=(5,5),activation='relu'))
model.add(AveragePooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
model.add(Dense(units=120,activation='relu'))
model.add(Dense(units=84,activation='relu'))
model.add(Dense(units=10,activation='softmax'))
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history=model.fit(trainX,trainy,batch_size=128,epochs=10,validation_data=(testX,testy))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
y_pred=np.argmax(model.predict(testX),axis=1)
for i in range(32,48):
 plt.subplot(4,4,i-31)
 plt.title(outputClass[y_pred[i]])
 plt.imshow(testX[i])
 plt.axis("off")
plt.show()
