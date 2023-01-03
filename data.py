import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

def mnist():
    # exchange with the corrupted mnist dataset
    
    trainImages = []
    path = "C:\\Users\\harid\\Desktop\\DTU_3rd_Semester\\MLOps\\dtu_mlops\\data\\corruptmnist\\"
    for i in os.listdir(path):
        if i.startswith("train"):
            data = np.load(path+i)
            trainImages.append(data)
        else:
            test = np.load(path+i)
    train = trainImages
    return train, test


trainImages = []
path = "C:\\Users\\harid\\Desktop\\DTU_3rd_Semester\\MLOps\\dtu_mlops\\data\\corruptmnist\\"
for i in os.listdir(path):
    if i.startswith("train"):
        data = np.load(path+i)
        trainImages.append(data)
    else:
        test = np.load(path+i)

# trimages = trainImages
# train = []
# for tr in trimages:
#     for key, arr in tr.items():
#         train.append(arr)


# for key, arr in train.items():
#     if key != "allow_pickle":
#         print(key, len(arr) )
#     if key == "labels":
#         print(np.unique(arr))
#         _ = plt.hist(arr, bins='auto')
#         plt.title("Histogram with 'auto' bins")
#         plt.show() 
train = []
import matplotlib.pyplot as plt
for tr in trainImages:
    for key, arr in tr.items():
        if key == "labels":
            train.extend(arr)

print(len(train))
_ = plt.hist(train, bins='auto')
plt.title("Histogram with 'auto' bins")
plt.show()  

# import matplotlib.pyplot as plt
# for tr in trainImages:
#     for key, arr in tr.items():
#         if key != "allow_pickle":
#             print(key, len(arr) )
#         if key == "labels":
#             print(np.unique(arr))
#             _ = plt.hist(arr, bins='auto')
#             plt.title("Histogram with 'auto' bins")
#             plt.show()  