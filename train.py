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

groundtruth = pd.read_excel('./trainingSet/groundTruth.xlsx', header=None)
groundtruth = np.array(groundtruth).flatten()

oriSamples = []
for idx in range(len(groundtruth)):
    img = Image.open('{0}/trainSample_{1:03d}.tiff'.format('./trainingSet', idx))
    img = img.resize((640, 480))
    imgArr = np.array(img)
    oriSamples.append(imgArr)

plt.imshow(oriSamples[0])

allSamples =np.array([img[:, :, np.newaxis] for img in artSamples(oriSamples)])
allTruths = keras.utils.to_categorical(np.repeat(groundtruth, 4))

normedSamples = (allSamples - allSamples.mean(0)) / allSamples.max()

model = Sequential()

model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(480, 640, 1)))
model.add(Conv2D(16, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# model.add(Conv2D(16, (3, 3), padding='same', input_shape=(960, 1280, 1)))
# # model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Conv2D(8, (3, 3), padding='same'))
# # model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.5))

# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dense(32))
# model.add(Activation('relu'))
# model.add(Dense(2))
# model.add(Activation('softmax'))

# model.summary()

model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['accuracy'])
model.summary()
print("Parameter number at {0}".format(model.count_params()))

history = model.fit(allSamples, allTruths, epochs=20, batch_size=8)
