import numpy as np
from PIL import Image
from io import BytesIO
from keras.preprocessing import image
import tensorflow as tf
tf.keras.utils.get_custom_objects()['silu'] = tf.nn.silu

# Patch mtcnn for compatibility with newer TensorFlow/Keras versions
# This is a workaround for the issue where mtcnn uses an old Keras API.
try:
    from mtcnn.mtcnn import MTCNN
    import inspect
    
    if 'weights_file' in inspect.getfullargspec(MTCNN.__init__).args:
        def patched_init(self, min_face_size=20, scale_factor=0.709, weights_file=None):
            # Original __init__ logic without weights_file
            self._min_face_size = min_face_size
            self._scale_factor = scale_factor
            self._steps_threshold = [0.6, 0.7, 0.7]

        MTCNN.__init__ = patched_init
except ImportError:
    # mtcnn not installed, or other issue.
    # The main error would have been raised elsewhere if it was critical.
    pass
# End of patch

# Define the target size expected by your Keras model
# Standard size for many facial models is 160x160 or 224x224
TARGET_SIZE = (160, 160) 

async def preprocess_image(file_content: bytes) -> np.ndarray:
    """Loads, resizes, and converts image bytes into a Keras-ready NumPy array."""
    
    # 1. Load the image from bytes
    img = Image.open(BytesIO(file_content))
    
    # 2. Resize the image
    img = img.resize(TARGET_SIZE)
    
    # 3. Convert to NumPy array
    img_array = image.img_to_array(img)
    
    # 4. Add batch dimension (1, H, W, C)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Normalize (Crucial step! Check your model's training process)
    # Common normalization for face models is 1./255. or mean subtraction
    img_array = img_array / 255.0  # Example: simple 0-1 scaling
    
    return img_array