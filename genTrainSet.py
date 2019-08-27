#%%
import numpy as np
from PIL import Image
from os import listdir
from shutil import copyfile

imageDir1 = 'D:/Snapshots/2019-08-23_Scanning/Scan2019-08-23 22.23.41'
imageDir2 = 'D:/Snapshots/2019-08-23_Scanning/Scan2019-08-25 22.27.41'

def copySamples(targetDir, shift):
    sampleList = ['{0}/{1}'.format(imageDir1, file) for file in listdir(imageDir1) if file.endswith('.tif')] + \
        ['{0}/{1}'.format(imageDir2, file) for file in listdir(imageDir2) if file.endswith('.tif')]

    randList = np.random.choice(sampleList, 40, False)

    for idx, file in enumerate(randList):
        copyfile(file, '{0}/trainSample_{1:03d}.tiff'.format(targetDir, idx + shift))
    
def artSamples(oriSamples):
    allSamples = []
    # oriSamples = []
    for imgArr in oriSamples:
        allSamples.append(imgArr)
        allSamples.append(np.flip(imgArr, 0))
        allSamples.append(np.flip(imgArr, 1))
        allSamples.append(np.rot90(np.rot90(imgArr)))

    return allSamples

#%%
copySamples('./trainingSet', 60)

#%%
