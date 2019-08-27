#%%
import numpy as np
import pandas as pd
import re
import argparse

from keras.models import Sequential, model_from_json

from os import listdir
from PIL import Image

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script to run the pre-trained model on a set of images, \
        imagees shoud end only with '.tif'.")
    parser.add_argument("WorkDir", default='./', help="The directory containing all the images (only ends with .tif)")

    args = parser.parse_args()
    wDir = args.WorkDir

    letter2num = dict(zip([chr(a) for a in range(65, 91)], (range(24))))
    modelRE = re.compile(r'modelArch_(\d+.\d+).json')
    normFactor = np.load('./savedModels/normFactors.npy')
    latestTS = 0
    latestTS_str = '0'
    for savedModel in listdir('./savedModels/'):
        if savedModel.endswith('.json'):
            TS_str = modelRE.match(savedModel).group(1)
            TS = float(TS_str)
            if TS > latestTS:
                latestTS = TS
                latestTS_str = TS_str
    with open('./savedModels/modelArch_{0}.json'.format(latestTS_str)) as f:
        latestModel = model_from_json(f.read())

    latestModel.load_weights('./savedModels/modelWeights_{0}.h5'.format(latestTS_str))
#%%
    wellPosRe = re.compile(r'\d{8}_[a-zA-Z0-9]+_([A-P])(\d{2})_1.tif')

    posList = []
    imgArrSet = []
    for imgFile in listdir(wDir):
        posMatch = wellPosRe.search(imgFile)
        if posMatch:
            wellPos = [posMatch.group(1), int(posMatch.group(2))]
            posList.append(wellPos)

            img = Image.open('{0}/{1}'.format(wDir, imgFile))
            imgArr = np.array(img)[:, :, np.newaxis]

            imgArrSet.append((imgArr - normFactor[0]) / normFactor[1])
    imgArrSet = np.array(imgArrSet)

#%%
    predictions = latestModel.predict(imgArrSet, verbose=True)
    predTable = np.zeros((16, 24)) - 1
    for pos, pred in zip(posList, predictions):
        predTable[letter2num[pos[0]], pos[1] - 1] = pred
    
    predDF = pd.DataFrame(predTable)
    predDF.to_excel('{0}/ML_Prediction_Model{1}.xlsx'.format(wDir, latestTS_str), header=False, index=False)

#%%
