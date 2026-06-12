from flask import Flask, render_template, request
from tensorflow import keras
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

model = keras.models.load_model("model_kopi.h5")
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