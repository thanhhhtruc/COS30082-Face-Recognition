import numpy as np
from PIL import Image
from io import BytesIO
from keras.preprocessing import image

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