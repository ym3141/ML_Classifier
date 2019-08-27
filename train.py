#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras import regularizers

from os import listdir
from genTrainSet import artSamples
from PIL import Image
from datetime import datetime

#%%
groundtruth = pd.read_excel('./trainingSet/groundTruth.xlsx', header=None)
groundtruth = np.array(groundtruth).flatten()

oriSamples = []
for idx in range(len(groundtruth)):
    img = Image.open('{0}/trainSample_{1:03d}.tiff'.format('./trainingSet', idx))
    img = img.resize((640, 480))
    imgArr = np.array(img)
    oriSamples.append(imgArr[:, :, np.newaxis])

allSamples =np.array([img for img in artSamples(oriSamples)])
allTruths = np.repeat(groundtruth, 4)

normedSamples = (allSamples - allSamples.mean(0)) / allSamples.max()

#%%
model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(480, 640, 1)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))    
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
print("Parameter number at {0}".format(model.count_params()))

history = model.fit(normedSamples, allTruths, epochs=30, batch_size=16, validation_split=0.2)

# model.evaluate


model.save('./savedModels/SavedModel_{0}.h5'.format(datetime.now().timestamp()))