import torch
import torchvision
from torchvision import transforms, utils
import torch.optim as optim
from torchvision import datasets
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from torch.utils.data import Dataset, DataLoader
import os
import sys
import time

# set the variables
IMGSIZE=300
IMG_RESIZE=300
IMGCHANNEL=3

torch.cuda.empty_cache()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ('cherry', 'strawberry', 'tomato')

train_dir = "traindata/"
test_dir = "testdata/"
infer_dir = "inferdata/"
TEST_BATCH_SIZE = 32

"""### Data Loader & Evaluate function"""

class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):

        img, label = super(ImageFolderWithPaths, self).__getitem__(index)

        path = self.imgs[index][0]
        img_size = img.shape

        return (img, label ,path)

transform = transforms.Compose([
        transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def f_inference(model, val_loader):

  total_pred = {classname: 0 for classname in CLASSES}
  l_preds = []
  step = int((EVAL_SIZE/TEST_BATCH_SIZE)/10) + 1
  for i, (data) in enumerate(val_loader, 0):
    X, _ = data
    with torch.no_grad():
      X = X.to(DEVICE)
      outputs = model(X)
      _, preds = torch.max(outputs, 1)
      preds = preds.to(DEVICE)
      print(preds)
      l_preds += preds.cpu().tolist()
    if i % step == 0:
      print(f'--- {len(l_preds)} images are evaluated.')


"""### Load model"""
load_model_file = 'model.pth'

if len(sys.argv) > 1:
  if (os.path.exists(sys.argv[1])):
    load_model_file = sys.argv[1]

if not(os.path.exists(load_model_file)):
  print("The default model does not exists")
  exit()

if len(sys.argv) > 2:
  test_dir = sys.argv[2]
  print(f"Test folder is set to {test_dir}")


enet =  torchvision.models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = enet.fc.in_features
enet.fc = nn.Linear(num_ftrs, 3)

enet.load_state_dict(torch.load(load_model_file, map_location=torch.device('cpu')))
print(f"Loaded model: {load_model_file}")

enet.eval()

"""### Evaluation Result"""

eval_set = ImageFolderWithPaths(test_dir, transform=transform) # add transformation directly
EVAL_SIZE = len(eval_set)

print(f"\nInference {EVAL_SIZE} images")

val_loader = DataLoader(eval_set, batch_size=TEST_BATCH_SIZE, shuffle=False) #EVAL_SIZE

start = time.time()
f_inference(enet, val_loader)

elapse = (time.time() - start)/60

print(f"\nElapse: {elapse:.1f} minutes")
