#%%
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint

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
    # img = img.resize((640, 480))
    imgArr = np.array(img)
    oriSamples.append(imgArr[:, :, np.newaxis])

allSamples =np.array([img for img in artSamples(oriSamples)])
allTruths = np.repeat(groundtruth, 4)

normedSamples = (allSamples - allSamples.mean(0)) / allSamples.max()

trainTruths = allTruths[0 : -100]
trainNormedSamples = normedSamples[0 : -100]

testTruths = allTruths[-100 : ]
testNormedSamples = normedSamples[-100 : ]

#%%
model = Sequential()

model.add(Conv2D(24, kernel_size=6, strides=3, activation='relu', input_shape=(960, 1280, 1)))
model.add(Conv2D(8, kernel_size=3, strides=2, activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(192, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))    
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
# print("Parameter number at {0}".format(model.count_params()))

timeStampCur = datetime.now().timestamp()
with open('./savedModels/modelArch_{0}.json'.format(timeStampCur), 'w+') as f:
    f.write(model.to_json())
np.save('./savedModels/normFactors_{0}.npy'.format(timeStampCur), np.array([allSamples.mean(0), allSamples.max()]))

earlyS = EarlyStopping(monitor='val_acc', patience=5, verbose=0, mode='max')
modelC = ModelCheckpoint('./savedModels/modelWeights_{0}.h5'.format(timeStampCur), 
                         save_best_only=True, monitor='val_acc', mode='max')

history = model.fit(trainNormedSamples, trainTruths, epochs=30, 
                    callbacks=[earlyS, modelC],
                    batch_size=16, validation_split=0.2)

loss, accu = model.evaluate(testNormedSamples, testTruths)
print('Evaluation result:\nLoss: {0:.5f}   Accuracy: {1:.3f}'.format(loss, accu))
