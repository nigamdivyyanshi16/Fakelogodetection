import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Canvas, Button
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Constants
DATASET_DIR = 'dataset'
GENUINE_DIR = os.path.join(DATASET_DIR, 'genuine')
FAKE_DIR = os.path.join(DATASET_DIR, 'fake')
MODEL_PATH = 'logo_authenticity_model.pkl'

# Data Collection and Feature Extraction
def extract_features_from_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (200, 200))  # Resize the image as needed
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Feature extraction (HOG)
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray).flatten()
    
    return features

def collect_data_and_train_model():
    X, y = [], []
    
    # Collect data from the 'genuine' folder
    for filename in os.listdir(GENUINE_DIR):
        if filename.endswith(".png"):
            features = extract_features_from_image(os.path.join(GENUINE_DIR, filename))
            X.append(features)
            y.append(1)  # Label 1 for genuine logos
    
    # Collect data from the 'fake' folder
    for filename in os.listdir(FAKE_DIR):
        if filename.endswith(".png"):
            features = extract_features_from_image(os.path.join(FAKE_DIR, filename))
            X.append(features)
            y.append(0)  # Label 0 for fake logos
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    # Test the model
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# Create a Tkinter GUI
root = tk.Tk()
root.title("Logo Authenticity Detection")
root.geometry("800x600")

def open_file():
    file_path = filedialog.askopenfilename(initialdir=DATASET_DIR)
    if file_path:
        try:
            features = extract_features_from_image(file_path)
            features = np.array(features).reshape(1, -1)
            features = StandardScaler().fit_transform(features)
            
            prediction = model.predict(features)
            
            if prediction[0] == 1:
                result_label.config(text="Genuine Logo")
            else:
                result_label.config(text="Fake Logo")
            
            image = Image.open(file_path)
            image = image.resize((400, 400), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(image=image)
            image_label.config(image=img)
            image_label.image = img
        except Exception as e:
            result_label.config(text="An error occurred")

browse_button = Button(root, text="Browse Image", command=open_file)
browse_button.pack()

result_label = Label(root, text="", font=("Helvetica", 18))
result_label.pack()

image_label = Label(root)
image_label.pack()

# Collect data, train the model, and test it
model, accuracy = collect_data_and_train_model()
print("Model Accuracy:", accuracy)

root.mainloop()
