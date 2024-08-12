import os
import base64
import tempfile
from flask import Flask, request, jsonify
from gradio_client import Client, file
from flask_cors import CORS
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({'message': 'Hello, world!'})

def decode_image(encoded_data):
    prefix, data = encoded_data.split(',', 1)
    image_data = base64.b64decode(data)
    format = prefix.split('/')[-1].split(';')[0]
    return Image.open(BytesIO(image_data)), format

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        json_data = request.get_json()
        face_image, face_format = decode_image(json_data['face_image'])

        with tempfile.NamedTemporaryFile(suffix=f'.{face_format}', delete=False) as img_file:
            face_image.save(img_file)
            image_path = img_file.name

        # Perform prediction using Gradio Client
        client = Client("https://booth.fineit.io/")
        result = client.predict(
            file(image_path),
            json_data.get('positive_prompt', ""),
            json_data.get('negative_prompt', "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green"),
            seed=-1,
            depth_map_feather_threshold=128,
            depth_map_dilation_iterations=10,
            depth_map_blur_radius=10,
            api_name="/generate"
        )

        # Delete temporary files
        os.remove(image_path)
        return jsonify({'result': result})
        prefix = 'data:image/png;base64,'
        generated_image_path = result[0]
        with open(generated_image_path, 'rb') as img_file:
            image_data = img_file.read()
        encoded_image = prefix + base64.b64encode(image_data).decode('utf-8')
        return jsonify({'image': encoded_image})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=7861)
