import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np  # Add this line
import tensorflow.compat.v1 as tf

new_model = tf.keras.models.load_model("models/CNN8020.h5")

# Define class names for prediction output
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Image path and preprocessing
img_path = 'C:/my files/IIUM/6/fyp_1/FYP/datasets/archive (5)/OCT2017/test/NORMAL/NORMAL-582215-6.jpeg'
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = img_batch / 255.0

# Define a TensorFlow function for prediction
@tf.function
def predict(model, image):
    return model(image)  # Return the predictions directly

# Make prediction
prediction = predict(new_model, img_preprocessed)
predicted_class = class_names[np.argmax(prediction[0])]

# Print the prediction result
print("Predicted class:", predicted_class)



model_files = [
    'CNN8020.h5',
    'my_resnet_model_T1.h5',
    'RS73.h5',
    'RS82.h5',
]