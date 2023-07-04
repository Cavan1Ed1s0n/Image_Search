from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.h5')
# See https://keras.io/api/applications/ for details

class FeatureExtractor:
    def __init__(self):
        self.model = Model(inputs=(model.input), outputs=model.get_layer('predictions').output)

    def extract(self,img):
        """
        Extract a deep feature from an input image
        Args:
            img: from PIL.Image.open(path) 

        Returns:
            feature (np.ndarray): deep feature with the shape=(4096, )
        """
        img = img.resize((224, 224)).convert('L')
        x = image.img_to_array(img)
        x=np.squeeze(x,axis=-1)
        x = np.stack((x,x,x),axis=-1)  
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), where the first elem is the number of img
        feature = self.model.predict(x)[0]  # (1, 4096) -> (4096, )
        return feature / np.linalg.norm(feature)  # Normalize
