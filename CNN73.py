#this to bulid CNN 70:30
# Import Dependencies
import os
import cv2
import imghdr
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
import pickle

# Optional GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Constants and Data Directories
data_dir_train = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\train'
data_dir_test = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\test'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']



# Getting and Preprocessing Data
def preprocess_data(data_dir):
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    data = data.map(lambda x, y: (x / 255, y))
    return data

train_data = preprocess_data(data_dir_train)
test_data = preprocess_data(data_dir_test)

# Display Sample Images (Optional)
def display_sample_images(data):
    batch = data.as_numpy_iterator().next()
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img)
        ax[idx].title.set_text(batch[1][idx])
    # plt.show()

display_sample_images(train_data)
print("test")

# Building the Deep Neural Network with Dropout
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')
])


# Compile the model
model.compile(optimizer='adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
model.summary()

# Train the Model
logdir = 'logs'
tensorboard_callback = TensorBoard(log_dir=logdir)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(train_data, epochs=20, validation_data=test_data, callbacks=[tensorboard_callback, early_stopping])

# Plot Performance
def plot_metric(history, metric, color, title):
    plt.figure()
    plt.plot(history.history[metric], color=color, label=metric)
    plt.plot(history.history[f'val_{metric}'], color='orange', label=f'val_{metric}')
    plt.title(title, fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

plot_metric(history, 'loss', 'teal', 'Loss')
plot_metric(history, 'accuracy', 'teal', 'Accuracy')

# Save model and history
model.save('CNN7030.h5')
with open('CNN7030.pkl', 'wb') as f:
    pickle.dump(history.history, f)
