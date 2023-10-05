import cv2
import numpy as np
import os

def preprocess_image(image_path, target_size=(224, 224)):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to load image at {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img / 255.0  # Normalize pixel values to the range [0, 1]
        return img
    except Exception as e:
        print(f"Error while preprocessing image at {image_path}: {str(e)}")
        return None

# Example usage:
image_path = 'train_data/genuine/adidasreal.png'
preprocessed_img = preprocess_image(image_path)

if preprocessed_img is not None:
    # Proceed with using the preprocessed image
    pass
else:
    # Handle the case where image preprocessing failed
    pass

