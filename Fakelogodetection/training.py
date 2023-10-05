import tensorflow as tf
from tensorflow.keras import layers, models

# Define paths
data_dir = 'dataset'
train_dir = 'train_data'  # Define 'train_dir' here


base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the pre-trained layers
base_model.trainable = False

# Add custom classification layers
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model using the training data
batch_size = 32
epochs = 10

train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
train_data = train_data_gen.flow_from_directory(train_dir,
                                                target_size=(224, 224),
                                                batch_size=batch_size,
                                                class_mode='binary')

history = model.fit(train_data, epochs=epochs)

# Save the trained model to a file
model.save('Fakedetectionmodel.keras')  # Replace 'your_trained_model.h5' with the desired model filename