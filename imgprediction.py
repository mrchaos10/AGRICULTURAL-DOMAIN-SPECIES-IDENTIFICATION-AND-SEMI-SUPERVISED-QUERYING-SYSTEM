import glob
import random

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm_notebook as tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F

import matplotlib.pyplot as plt
import pickle

from sklearn.svm import SVC 
from sklearn.naive_bayes import BernoulliNB 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def get_profile_path(category):

    data = []

    for path in sorted(glob.glob('C:/Users/Varadharajan R/Desktop/FYP/img/*.jpg')):
        #path=re.sub(r" ?\([^)]+\)","",path)
        
        
        label=str(path.split('/')[-1].split('.')[0]).replace(category,'')[1:]
        label = ''.join(i for i in label if not i.isdigit())

        data.append({
			'path' : path,
            'label': label ,
        })
            
    return pd.DataFrame(data)

train = get_profile_path('train')
print(train)


def resize_to_square(image, size):
    h, w, d = image.shape
    ratio = size / max(h, w)
    resized_image = cv2.resize(image, (int(w*ratio), int(h*ratio)), cv2.INTER_AREA)
    return resized_image

def image_to_tensor(image, normalize=None):
    tensor = torch.from_numpy(np.moveaxis(image / (255. if image.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
    if normalize is not None:
        return F.normalize(tensor, **normalize)
    return tensor

def pad(image, min_height, min_width):
    h,w,d = image.shape

    if h < min_height:
        h_pad_top = int((min_height - h) / 2.0)
        h_pad_bottom = min_height - h - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if w < min_width:
        w_pad_left = int((min_width - w) / 2.0)
        w_pad_right = min_width - w - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    return cv2.copyMakeBorder(image, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, cv2.BORDER_CONSTANT, value=(0,0,0))


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, df, size):
        self.df = df
        self.size = size
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = cv2.imread(row.path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_to_square(image, self.size)
        image = pad(image, self.size, self.size)
        tensor = image_to_tensor(image, normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]})
            
        return tensor


random.seed(70)
		
size = 224




#densenet model with features layer
model_name = 'densenet121'
layer_name = 'features'

#resnet model with average pool layer
# model_name = 'resnet18'
# layer_name = 'avgpool'

get_model = getattr(torchvision.models, model_name)

def extract_features(datf):

    model = get_model(pretrained=True)
    model = model.cuda()
    model.eval()

    # register hook to access to features in forward pass
    features = []
    def hook(module, input, output):
        N,C,H,W = output.shape
        output = output.reshape(N,C,-1)
        features.append(output.mean(dim=2).cpu().detach().numpy())
    handle = model._modules.get(layer_name).register_forward_hook(hook)

    dataset = Dataset(datf, size)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    for i_batch, inputs in tqdm(enumerate(loader), total=len(loader)):
        _ = model(inputs.cuda())

    features = np.concatenate(features)

    features = pd.DataFrame(features)
    features = features.add_prefix('PIXEL_')
    features.loc[:,'label'] = datf['label']
    features.loc[:,'path'] = datf['path']
 
    
    
    handle.remove()
    del model

    return features

features_train = extract_features(train)
cols = features_train.select_dtypes(include=['float32','int32']).columns
features_train = pd.DataFrame(features_train, columns = cols)

#print(features_train)


svm_model=pickle.load(open('Models_densenet/SVM.pkl','rb'))
dtc_model=pickle.load(open('Models_densenet/DTC.pkl','rb'))
nb_model=pickle.load(open('Models_densenet/naive_bayes.pkl','rb'))

svm_result=svm_model.predict(features_train)
dtc_result=dtc_model.predict(features_train)
nb_result=nb_model.predict(features_train)

print("SVM LABEL : ",svm_result)
print("DTC LABEL : ",dtc_result)
print("NB LABEL : ",nb_result)
