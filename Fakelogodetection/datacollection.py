import os
import shutil

# Define paths
data_dir = 'dataset'
train_dir = 'train_data'
test_dir = 'test_data'

# Split the data into training and testing sets (e.g., 80% for training, 20% for testing)
split_ratio = 0.8

# Create directories for training and testing data if they don't exist
os.makedirs(os.path.join(train_dir, 'genuine'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'fake'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'genuine'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'fake'), exist_ok=True)


# Collect genuine logos
genuine_images = os.listdir(os.path.join(data_dir, 'genuine'))
num_genuine = len(genuine_images)

# Copy 80% of genuine logos to the training set
split_index = int(split_ratio * num_genuine)
for i in range(split_index):
    src_path = os.path.join(data_dir, 'genuine', genuine_images[i])
    dst_path = os.path.join(train_dir, 'genuine', genuine_images[i])
    shutil.copy(src_path, dst_path)

# Copy the remaining 20% to the test set
for i in range(split_index, num_genuine):
    src_path = os.path.join(data_dir, 'genuine', genuine_images[i])
    dst_path = os.path.join(test_dir, 'genuine', genuine_images[i])
    shutil.copy(src_path, dst_path)

# Collect fake logos and perform the same split for training and testing
fake_images = os.listdir(os.path.join(data_dir, 'fake'))
num_fake = len(fake_images)

split_index = int(split_ratio * num_fake)
for i in range(split_index):
    src_path = os.path.join(data_dir, 'fake', fake_images[i])
    dst_path = os.path.join(train_dir, 'fake', fake_images[i])
    shutil.copy(src_path, dst_path)

for i in range(split_index, num_fake):
    src_path = os.path.join(data_dir, 'fake', fake_images[i])
    dst_path = os.path.join(test_dir, 'fake', fake_images[i])
    shutil.copy(src_path, dst_path)
