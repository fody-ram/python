from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
import io
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = load_model("models/CNN8020.h5")
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

@app.route("/classify", methods=["POST"])
def classify_image():
    data = request.get_json()
    image_data = data['image']
    try:
        decoded_data = base64.b64decode(image_data)
        img = image.load_img(io.BytesIO(decoded_data), target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_batch = np.expand_dims(img_array, axis=0)
        img_preprocessed = img_batch / 255.0
        prediction = model.predict(img_preprocessed)
        predicted_class = class_names[np.argmax(prediction[0])]
        return jsonify({"predicted_class": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/test")
def test_connection():
    return "Connection successful!"

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)