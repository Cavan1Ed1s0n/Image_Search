## Extract feature (old model)
Use file `feature_extractor.py` to transform image to feature vector length 4096. The vector can be stored in file `.npy`. 
You should read image before input feature_extract:
```python
from PIL import Image
fe = FeatureExtractor()
feature = fe.extract(Image.open(img_path))
```
In file `feature_extractor.py` you must load model. You can download model from [this](https://drive.google.com/file/d/15tOrpFsFTCynGVoUi5PH5Pl6n94WzYX0/view?usp=sharing)

Folder [model](https://drive.google.com/file/d/1ruUAEWDOsrU9w74tnDP40AxvcLWyR6yp/view?usp=sharing) in drive 

Use this code below to extract and save features of images.
```python
from PIL import Image
from feature_extractor import FeatureExtractor
from pathlib import Path
import numpy as np
from tqdm import tqdm
fe = FeatureExtractor()

for img_path in tqdm(sorted(Path("./images").glob("*.jpg"))):
    feature = fe.extract(img=Image.open(img_path))
    feature_path = Path("./Image_Search/feature_model_cnn") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
    np.save(feature_path, feature)
```

## Extract feature (new model with pytorch)

Download [model](https://drive.google.com/file/d/1Sk71Avnl-PZ5RIWPJGsnUyTD7QgXiKV5/view?usp=sharing).
```python
import pandas as pd
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm


import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from pathlib import Path

#load model
def load_model(epoch, model, optimizer):
    """Loads model state from file.
    """
    file = Path.cwd() / f'model_{len(idx_to_class)}_checkpoints' / f'epoch_{epoch}.pt'
    checkpoint = torch.load(file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

model,optimizer = load_model(1000,model,optimizer)

# Read image features
extract_features = []
img_paths = []
for feature_path in Path("./feature_finetuned_2000").glob("*.npy"):    
    extract_features.append(np.load(feature_path))
    img_paths.append(Path("./images_part1") / (feature_path.stem + ".jpg"))

#input image
file = "images_part1/0008e7ec357f64b1e05a123b85b105492ac975a2.jpg"
img = Image.open(file)  # PIL image
image_ = img.copy()
# Run search
img = image.img_to_array(img)
img = test_transforms(image=img)['image'].to(device)
img = torch.unsqueeze(img, 0)
# model= torch.load(f'model_{len(idx_to_class)}_finetuned_2000') 
features={}

#get features of input
def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
model.module.fc1.register_forward_hook(get_features('feats'))

out = model(img)
query = features['feats'].cpu().numpy()
dists = np.linalg.norm((extract_features-query).squeeze(), axis=1)  # L2 distances to features
ids = np.argsort(dists)[:5]  # Top 5 results
scores = [(dists[id], img_paths[id]) for id in ids]

#visualise performance
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 3
fig.add_subplot(rows, columns, 1)
plt.imshow(image_)
plt.axis('off')
plt.title("Sample")
for i in tqdm(range(len(scores))):
    imagepath = scores[i][1]
    Image = cv2.cvtColor(cv2.imread(str(imagepath)), cv2.COLOR_RGB2BGR)
    fig.add_subplot(rows, columns, i+2)
    plt.imshow(Image)
    plt.axis('off')
    plt.title(str(scores[i][0]))
    

```
