## Extract feature 
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
