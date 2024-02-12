import torch
import torchvision
from torchvision import transforms, utils
import torch.optim as optim
from torchvision import datasets
import math
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from PIL import Image
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


transform = transforms.Compose([
        transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def f_inference(model, img_tensor):
  with torch.no_grad():
    X = img_tensor.to(DEVICE)
    outputs = model(X)
    _, preds = torch.max(outputs, 1)
    preds = preds.to(DEVICE)

    return(preds.item())

dir_path = os.path.dirname(os.path.realpath(__file__))
print(f"Working folder: {dir_path}")

"""### Load model"""
load_model_file = dir_path + '/model.pth'
infer_dir = dir_path + "/inferdata"
if len(sys.argv) > 1:
  if (os.path.exists(sys.argv[1])):
    load_model_file = sys.argv[1]
  elif (os.path.exists(dir_path + '/' + sys.argv[1])):
    load_model_file = dir_path + '/' + sys.argv[1]

if not(os.path.exists(load_model_file)):
  print("The default model does not exists")
  exit()

if len(sys.argv) > 2:
  if (os.path.exists(sys.argv[2])):
    infer_dir = sys.argv[2]
  elif (os.path.exists(dir_path + '/' + sys.argv[2])):
    infer_dir = dir_path + '/' + sys.argv[2]
  print(f"Test folder: {infer_dir}")

if not(os.path.exists(infer_dir)):
  print("The inference folder does not exists")
  exit()

enet =  torchvision.models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = enet.fc.in_features
enet.fc = nn.Linear(num_ftrs, 3)

enet.load_state_dict(torch.load(load_model_file, map_location=torch.device('cpu')))
print(f"Loaded model: {load_model_file}")

enet.eval()

"""### Evaluation Result"""

# %matplotlib inline
# import matplotlib.pyplot as plt

annotations = []
images = glob(infer_dir + '/*.jpg')
EVAL_SIZE = len(images)
print(f"\nInference {EVAL_SIZE} images")

max_row_plot = 10
plots_per_row = 5
row_plot = int(np.ceil(EVAL_SIZE/plots_per_row))
if row_plot > max_row_plot: print('Subset of results are plotted')

nr_plot = min(row_plot, max_row_plot)
fig = plt.figure(figsize=(20, 4*nr_plot))
for i, image_fn in enumerate(images):
  img = Image.open(image_fn)
  img_tensor = transform(img).unsqueeze(0)

  preds=f_inference(enet, img_tensor)
  print(image_fn, "|" , CLASSES[preds])

  
  ax = fig.add_subplot(nr_plot, plots_per_row, i+1)
  ax.imshow(img)
  ax.axis('off')
  ax.set_title('Predict: ' + CLASSES[preds])
  if (i+1) / plots_per_row == nr_plot: break
plt.savefig('inference_result.png')
print('Inference result is save at ', dir_path, '/inference_result.png')

