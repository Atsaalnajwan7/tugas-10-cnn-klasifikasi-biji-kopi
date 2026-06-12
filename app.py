from flask import Flask, render_template, request
from tensorflow import keras
import numpy as np
from PIL import Image
import os
import gdown

app = Flask(__name__)

MODEL_PATH = "model_kopi.h5"
FILE_ID = "1ctXjkMnl27NhCf3Akt1yDDfkECiAGJny"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
    print("Downloading model from Google Drive...")
    gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

model = keras.models.load_model(MODEL_PATH)
class_names = ['Dark', 'Green', 'Light', 'Medium']

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            img = Image.open(filepath).resize((224, 224)).convert('RGB')
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            pred = model.predict(img_array)
            predicted_index = np.argmax(pred)
            prediction = class_names[predicted_index]
            confidence = round(float(np.max(pred)) * 100, 2)
            image_path = filepath

    return render_template('index.html', prediction=prediction,
                            confidence=confidence, image=image_path)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)