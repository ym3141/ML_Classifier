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

#%%
groundtruth = pd.read_excel('./trainingSet/groundTruth.xlsx', header=None)
groundtruth = np.array(groundtruth).flatten()

oriSamples = []
for idx in range(len(groundtruth)):
    imgArr = np.array(Image.open('{0}/trainSample_{1:03d}.tiff'.format('./trainingSet', idx)))
    oriSamples.append(imgArr)

allSamples =np.array([img[:, :, np.newaxis] for img in artSamples(oriSamples)])
allTruths = np.repeat(groundtruth, 4)



#%%
model = Sequential()
model.add(Conv2D(16, (3, 3), padding='same', input_shape=(960, 1280, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(8, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print("Parameter number at {0}".format(model.count_params()))
#%%
history = model.fit(allSamples, allTruths, epochs=10, batch_size=16)


#%%
