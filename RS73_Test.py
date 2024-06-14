#this is to test resnet_model_T1
# Imports
import os
import numpy as np
import cv2
import imghdr
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle  # Added for loading history

# Optional GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Constants
data_dir_test = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\test'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

# Getting Data
data_test = tf.keras.utils.image_dataset_from_directory(data_dir_test)
data_iterator_test = data_test.as_numpy_iterator()

# Preprocessing Data
data_test = data_test.map(lambda x, y: (x/255, y))

# Load the model
new_model = tf.keras.models.load_model("RS50_70_30_T1.h5")

# Load the history object
with open('RS50_70_30_history_T1.pkl', 'rb') as f:
    history = pickle.load(f)

# Plot Performance
def plot_history(history, title, label1, label2, color1='teal', color2='orange'):
    fig = plt.figure()
    plt.plot(history[label1], color=color1, label=label1)
    plt.plot(history[label2], color=color2, label=label2)
    fig.suptitle(title, fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

plot_history(history, 'Loss', 'loss', 'val_loss')
plot_history(history, 'Accuracy', 'accuracy', 'val_accuracy')

# Evaluate
true_labels = []
predicted_labels = []

for batch in data_test.as_numpy_iterator():
    X, y = batch
    yhat = new_model.predict(X)
    yhat_classes = yhat.argmax(axis=1)

    if len(y.shape) > 1:
        true_labels.extend(np.argmax(y, axis=1))
    else:
        true_labels.extend(y)

    predicted_labels.extend(yhat_classes)

accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='weighted')
recall = recall_score(true_labels, predicted_labels, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
