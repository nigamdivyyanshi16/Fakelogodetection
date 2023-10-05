import tkinter as tk
from tkinter import filedialog, Label, Canvas
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk

# Constants
DATASET_DIR = 'dataset'
GENUINE_DIR = os.path.join(DATASET_DIR, 'genuine')
FAKE_DIR = os.path.join(DATASET_DIR, 'fake')
MODEL_PATH = 'Fakedetectionmodel.keras'  # Replace with your trained model path

# Load the pre-trained CNN model for logo detection
model = load_model(MODEL_PATH)

# Create a function to preprocess and classify an image
def classify_image(file_path):
    try:
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (400, 400))

        # Extract HOG features from the image
        hog = cv2.HOGDescriptor()
        features = hog.compute(img).flatten()
        features = features.reshape(1, -1)

        # Predict using the pre-trained model
        prediction = model.predict(features)

        if prediction < 0.5:
            result_label.config(text="Genuine Logo")
        else:
            result_label.config(text="Fake Logo")

        # Display the selected image on the Tkinter window with a bounding box
        image = Image.fromarray(img)
        image = ImageTk.PhotoImage(image=image)
        image_label.config(image=image)
        image_label.image = image


        # Clear previous bounding boxes on the canvas
        # Clear previous bounding boxes on the canvas
        try:
            canvas.delete("all")
        except Exception as e:
            result_label.config(text="An error occurred: " + str(e))
    except Exception as e:
        result_label.config(text="An error occurred: " + str(e))

# Create the GUI window
root = tk.Tk()
root.title("Logo Authenticity Detection")
root.geometry("800x600")  # Set the window size

# Create a button to open an image file
def open_file():
    file_path = filedialog.askopenfilename(initialdir=DATASET_DIR)
    if file_path:
        classify_image(file_path)

browse_button = tk.Button(root, text="Browse Image", command=open_file)
browse_button.pack()

# Create a label to display the authenticity result
result_label = tk.Label(root, text="", font=("Helvetica", 18))
result_label.pack()

# Create a canvas to display the selected image with a bounding box
canvas = Canvas(root, width=400, height=400)
canvas.pack()

root.mainloop()
