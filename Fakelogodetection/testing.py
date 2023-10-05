import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load your trained model
model = load_model('Fakedetectionmodel.keras')  # Replace 'your_trained_model.h5' with the actual path to your saved model

# Define the directory containing the test images
test_data_dir = 'test_data'  # Change this to the path where your test data is located

# Create a data generator for test data
test_data_gen = ImageDataGenerator(rescale=1.0/255.0)
batch_size = 32  # Adjust this based on your preferences and available memory

test_data = test_data_gen.flow_from_directory(test_data_dir,
                                              target_size=(224, 224),  # Adjust the target size if needed
                                              batch_size=batch_size,
                                              class_mode='binary',  # Adjust for your dataset's class mode
                                              shuffle=False)  # Keep the order of images

# Perform inference on the test data
predictions = model.predict(test_data)

# Convert predictions to class labels (0 or 1)
predicted_labels = np.round(predictions).flatten()

# Get true labels from the test data generator
true_labels = test_data.classes

# Calculate accuracy
accuracy = np.mean(np.equal(true_labels, predicted_labels))

print(f"Test accuracy: {accuracy * 100:.2f}%")

