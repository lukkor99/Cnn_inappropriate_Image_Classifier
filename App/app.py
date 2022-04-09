from flask import Flask, request, jsonify
from app_utils import transform, get_predict

# app startup
app = Flask(__name__)
EXTENSIONS = {'png', 'jpg', 'jpeg'}


# file preprocessing
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONS


# predicting method
@app.route('/predict_image', methods=['POST'])
def predict_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'error':'format not supported'})

        try:
            img_bytes = file.read()
            tensor = transform(img_bytes)
            pred = get_predict(tensor)
            data = {'prediction': pred.item(), 'class name': str(pred.item())}
            return jsonify(data)
        except Exception as e:
            return jsonify({'error':e})
