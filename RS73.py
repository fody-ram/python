# this to bulid resnet_model_T1
# Imports
import os
import pickle
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# Optional GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Constants and data directories
data_dir_train = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\train'
data_dir_test = 'C:\\my files\\IIUM\\6\\fyp_1\\FYP\\datasets\\archive (5)\\OCT2017\\test'
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Getting and Preprocessing Data
train_data = tf.keras.utils.image_dataset_from_directory(data_dir_train)
test_data = tf.keras.utils.image_dataset_from_directory(data_dir_test)

train_data = train_data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=len(class_names))))
test_data = test_data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=len(class_names))))

# Model Definition
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
logdir = 'logs'
tensorboard_callback = TensorBoard(log_dir=logdir)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Training the model
history = model.fit(train_data, epochs=20, validation_data=test_data, callbacks=[tensorboard_callback, early_stopping])

# Plot performance
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
model.save('RS50_70_30_T1.h5')
with open('RS50_70_30_history_T1.pkl', 'wb') as f:
    pickle.dump(history.history, f)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
