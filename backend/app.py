from flask import Flask, request, send_file
from flask_cors import CORS
import os
from generator import generate_dst_from_image  # Updated function name

app = Flask(__name__)       # ✅ MUST come before @app.route
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])  # ✅ Now app is defined
def upload_image():
    if 'file' not in request.files:
        return {'error': 'No file provided'}, 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    dst_path = os.path.join(OUTPUT_FOLDER, 'output.dst')
    success = generate_dst_from_image(filepath, dst_path)

    if not success:
        return {'error': 'DST generation failed'}, 500

    return send_file(dst_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
