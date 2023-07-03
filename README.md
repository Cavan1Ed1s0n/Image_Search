## Extract feature 
Use file `feature_extractor.py` to transform image to feature vector length 4096. The vector can be stored in file `.npy`. 
You should read image before input feature_extract:
```python
from PIL import Image
fe = FeatureExtractor()
feature = fe.extract(Image.open(img_path))
```
In file `feature_extractor.py` you must load model. You can download model from [this](https://drive.google.com/file/d/15tOrpFsFTCynGVoUi5PH5Pl6n94WzYX0/view?usp=sharing)
