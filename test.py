import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # nopep8
warnings.filterwarnings("ignore", category=UserWarning)  # nopep8
import os
import tempfile
import math
from tqdm import tqdm
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import binary_dilation
import numpy as np
from gradio_client import Client, file
from captioner import init as init_captioner, derive_caption
from upscaler import init as init_upscaler
from segmenter import init as init_segmenter, segment
from depth_estimator import init as init_depth_estimator, get_depth_map
from pipeline import init as init_pipeline, run_pipeline
from image_utils import ensure_resolution, crop_centered
from flask import Flask, request, jsonify, send_file
import base64
from io import BytesIO
from PIL import Image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return jsonify({'api': {0:'/generate_image',1:'/replace_background'}})

developer_mode = os.getenv('DEV_MODE', False)

init_captioner()
init_upscaler()
init_segmenter()
init_depth_estimator()
init_pipeline()

torch.cuda.empty_cache()

POSITIVE_PROMPT_SUFFIX = "commercial product photography, 24mm lens f/8"
NEGATIVE_PROMPT_SUFFIX = "cartoon, drawing, anime, semi-realistic, illustration, painting, art, text, greyscale, (black and white), lens flare, watermark, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, floating, levitating"

MEGAPIXELS = 1.0

def decode_image(encoded_data):
    prefix, data = encoded_data.split(',', 1)
    image_data = base64.b64decode(data)
    format = prefix.split('/')[-1].split(';')[0]
    return Image.open(BytesIO(image_data)), format

@app.route('/generate_image/', methods=['POST'])
def generate_image():
    try:
        json_data = request.get_json()
        face_image, face_format = decode_image(json_data['face_image'])
        pose_image, pose_format = None, None

        pose_data = json_data.get('pose_image')
        if pose_data:
            pose_image, pose_format = decode_image(pose_data)

        with tempfile.NamedTemporaryFile(suffix=f'.{face_format}', delete=False) as img_file:
            face_image.save(img_file)
            image_path = img_file.name

        pose_path = None
        if pose_image:
            with tempfile.NamedTemporaryFile(suffix=f'.{pose_format}', delete=False) as img_file:
                pose_image.save(img_file)
                pose_path = img_file.name

        # Perform prediction using Gradio Client
        client = Client("https://booth.fineit.io/")
        result = client.predict(
            file(image_path),
            None if not pose_image else file(pose_path),
            json_data.get('prompt', ""),
            json_data.get('negative_prompt', "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green"),
            json_data.get('style_name', "(No style)"),
            json_data.get('num_steps', 30),
            json_data.get('identitynet_strength_ratio', 0.8),
            json_data.get('adapter_strength_ratio', 0.8),
            json_data.get('pose_strength', 0.4),
            json_data.get('canny_strength', 0.4),
            json_data.get('depth_strength', 0.4),
            json_data.get('controlnet', ["pose"]),
            json_data.get('guidance_scale', 2),
            json_data.get('seed', 1741991570),
            json_data.get('schedulers', "EulerDiscreteScheduler"),
            json_data.get('enable_fast_inference', False),
            json_data.get('enhance_non_face_region', True),
            api_name="/generate_image"
        )

        # Delete temporary files
        os.remove(image_path)
        if pose_path:
            os.remove(pose_path)

        prefix = 'data:image/png;base64,'
        generated_image_path = result[0]
        with open(generated_image_path, 'rb') as img_file:
            image_data = img_file.read()
        encoded_image = prefix + base64.b64encode(image_data).decode('utf-8')
        return jsonify({'image': encoded_image})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/replace_background/', methods=['POST'])
def replace_background_api():
    data = request.json
    
    # Get positive_prompt and negative_prompt from the request data
    positive_prompt = data.get('positive_prompt')
    negative_prompt = data.get('negative_prompt')
    options = {
        'seed': -1,
        'depth_map_feather_threshold': 128,
        'depth_map_dilation_iterations': 10,
        'depth_map_blur_radius': 10,
    }
    
    # Decode the base64 encoded image
    encoded_image = data.get('image')
    if encoded_image:
        image, face_format = decode_image(encoded_image)

        # image = Image.open(BytesIO(image_data))
        image = image.convert("RGB")
        print(image)

        result = replace_background(image, positive_prompt, negative_prompt, options)
        print(result[0])
        encoded_images = {}
        for i, img in enumerate(result[0][:4], start=1):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            img_str_with_prefix = 'data:image/png;base64,' + img_str
            encoded_images[f"image{i}"] = img_str_with_prefix

        return jsonify(encoded_images)
    else:
        return jsonify({'error': 'Image data not provided'}), 400

    return jsonify("response_data")

def replace_background(
    original,
    positive_prompt,
    negative_prompt,
    options,
):
    pbar = tqdm(total=7)

    print("Original size:", original.size)

    print("Captioning...")
    caption = derive_caption(original)
    pbar.update(1)

    print("Caption:", caption)

    torch.cuda.empty_cache()

    print(f"Ensuring resolution ({MEGAPIXELS}MP)...")
    resized = ensure_resolution(original, megapixels=MEGAPIXELS)
    pbar.update(1)

    print("Resized size:", resized.size)

    torch.cuda.empty_cache()

    print("Segmenting...")
    [cropped, crop_mask] = segment(resized)
    pbar.update(1)

    torch.cuda.empty_cache()

    print("Depth mapping...")
    depth_map = get_depth_map(resized)
    pbar.update(1)

    torch.cuda.empty_cache()

    print("Feathering the depth map...")

    # Convert crop mask to grayscale and to numpy array
    crop_mask_np = np.array(crop_mask.convert('L'))

    # Convert to binary and dilate (grow) the edges
    # adjust threshold as needed
    crop_mask_binary = crop_mask_np > options.get(
        'depth_map_feather_threshold')
    # adjust iterations as needed
    dilated_mask = binary_dilation(
        crop_mask_binary, iterations=options.get('depth_map_dilation_iterations'))

    # Convert back to PIL Image
    dilated_mask = Image.fromarray((dilated_mask * 255).astype(np.uint8))

    # Apply Gaussian blur and normalize
    dilated_mask_blurred = dilated_mask.filter(
        ImageFilter.GaussianBlur(radius=options.get('depth_map_blur_radius')))
    dilated_mask_blurred_np = np.array(dilated_mask_blurred) / 255.0

    # Normalize depth map, apply blurred, dilated mask, and scale back
    depth_map_np = np.array(depth_map.convert('L')) / 255.0
    masked_depth_map_np = depth_map_np * dilated_mask_blurred_np
    masked_depth_map_np = (masked_depth_map_np * 255).astype(np.uint8)

    # Convert back to PIL Image
    masked_depth_map = Image.fromarray(masked_depth_map_np).convert('RGB')

    pbar.update(1)

    final_positive_prompt = f"{caption}, {positive_prompt}, {POSITIVE_PROMPT_SUFFIX}"
    final_negative_prompt = f"{negative_prompt}, {NEGATIVE_PROMPT_SUFFIX}"

    print("Final positive prompt:", final_positive_prompt)
    print("Final negative prompt:", final_negative_prompt)

    print("Generating...")

    generated_images = run_pipeline(
        positive_prompt=final_positive_prompt,
        negative_prompt=final_negative_prompt,
        image=[masked_depth_map],
        seed=options.get('seed')
    )
    pbar.update(1)

    torch.cuda.empty_cache()

    print("Compositing...")

    composited_images = [
        Image.alpha_composite(
            generated_image.convert('RGBA'),
            crop_centered(cropped, generated_image.size)
        ) for generated_image in generated_images
    ]
    pbar.update(1)
    pbar.close()

    print("Done!")

    if developer_mode:
        pre_processing_images = [
            [resized, "Resized"],
            [crop_mask, "Crop mask"],
            [cropped, "Cropped"],
            [depth_map, "Depth map"],
            [dilated_mask, "Dilated mask"],
            [dilated_mask_blurred, "Dilated mask blurred"],
            [masked_depth_map, "Masked depth map"]
        ]
        return [
            composited_images,
            generated_images,
            pre_processing_images,
            caption,
        ]
    else:
        return [composited_images, None, None, None]


if __name__ == '__main__':
    app.run(port=7861)