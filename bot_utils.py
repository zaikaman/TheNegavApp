import os
import base64
import requests
from PIL import Image, ImageDraw
import random
import shutil
from gradio_client import Client, handle_file
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import io

# Hàm kiểm tra xem username đã được authenticated chưa
def is_user_authenticated(username):
    if not os.path.exists("authenticated_users.txt"):
        return False
    with open("authenticated_users.txt", "r") as file:
        authenticated_users = file.read().splitlines()
    return username in authenticated_users

# Hàm lưu username vào file authenticated_users.txt
def save_authenticated_user(username):
    with open("authenticated_users.txt", "a") as file:
        file.write(f"{username}\n")

# Converts an image file to Base64
def to_b64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Saves a Base64 encoded image to a file
def save_base64_image(base64_string, output_path):
    with open(output_path, "wb") as out_file:
        out_file.write(base64.b64decode(base64_string))

# Function to generate mask using SAM API with a fallback
def generate_mask(input_image_path):
    primary_client = Client("SkalskiP/florence-sam-masking")
    secondary_client = Client("tb2l/florence-sam-masking")

    try:
        # Attempt to generate mask using primary API
        result = primary_client.predict(
            image_input=handle_file(input_image_path),
            text_input="clothing",
            api_name="/process_image"
        )

        # Print the result to inspect the response
        print(f"Primary API Response: {result}")

        # Check if the result contains a valid file path
        if result and isinstance(result, str) and os.path.exists(result):
            mask_image_path = result  # This is the full path to the generated mask image
            print(f"Mask generated at: {mask_image_path}")

            # Define the path to save the mask in the working directory
            mask_save_path = os.path.join(os.getcwd(), "mask.jpg")

            # Copy the generated mask from temp to the current working directory
            shutil.copy(mask_image_path, mask_save_path)
            print(f"Mask saved as: {mask_save_path}")
            return mask_save_path
        else:
            print("Primary API: Mask generation failed. Trying secondary API.")
            return fallback_generate_mask(input_image_path, secondary_client)

    except Exception as e:
        print(f"Primary API Error generating mask: {str(e)}")
        print("Trying secondary API.")
        return fallback_generate_mask(input_image_path, secondary_client)

# Function to attempt mask generation using the secondary API
def fallback_generate_mask(input_image_path, client):
    try:
        result = client.predict(
            image_input=handle_file(input_image_path),
            text_input="clothing",
            api_name="/process_image"
        )

        # Print the result to inspect the response
        print(f"Secondary API Response: {result}")

        if result and isinstance(result, str) and os.path.exists(result):
            mask_image_path = result  # This is the full path to the generated mask image
            print(f"Mask generated at (fallback): {mask_image_path}")

            # Define the path to save the mask in the working directory
            mask_save_path = os.path.join(os.getcwd(), "mask.jpg")

            # Copy the generated mask from temp to the current working directory
            shutil.copy(mask_image_path, mask_save_path)
            print(f"Mask saved as (fallback): {mask_save_path}")
            return mask_save_path
        else:
            print("Error: Mask generation failed in secondary API. Invalid result or file does not exist.")
            return None

    except Exception as e:
        print(f"Error generating mask in secondary API: {str(e)}")
        return None

# Inpainting function with logging for input and mask
async def inpaint(input_image_path, mask_image_path, output_image_path):
    api_key = "SG_b77a34429a1aeb2e"
    url = "https://api.segmind.com/v1/sd1.5-inpainting"

    # Generate a random seed
    random_seed = random.randint(0, 2**32 - 1)

    # Default prompt
    prompt = "undress, naked"

    # Convert both input and mask image to base64
    input_image_b64 = to_b64(input_image_path)
    mask_image_b64 = to_b64(mask_image_path)

    print(f"Sending inpainting request with random seed: {random_seed}")
    print(f"Input image length: {len(input_image_b64)}, Mask image length: {len(mask_image_b64)}")

    data = {
        "prompt": prompt,
        "negative_prompt": "blurry, badquality, lowquality, sketches, clothing, underwear, bra, t-shirt, shirt, dress, skirt, clothes",
        "samples": 1,
        "image": input_image_b64,  # Send input image base64
        "mask": mask_image_b64,    # Send mask image base64
        "scheduler": "DDIM",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "strength": 1,
        "seed": random_seed,  # Randomized seed
        "img_width": 512,
        "img_height": 512
    }

    headers = {'x-api-key': api_key}

    try:
        # Send POST request to the API
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Check if the response contains a base64 encoded image
        try:
            response_json = response.json()
            output_image_b64 = response_json.get("image")

            if output_image_b64:
                print(f"Received inpainting response, image length: {len(output_image_b64)}")
                return save_base64_image(output_image_b64, output_image_path)
            else:
                print("Error: Image key not found in response JSON.")
                return None

        except requests.exceptions.JSONDecodeError:
            # If the response is not JSON, check if it's an image
            if response.headers['Content-Type'] == 'image/jpeg':
                with open(output_image_path, 'wb') as img_file:
                    img_file.write(response.content)
                print(f"Image saved directly from response: {output_image_path}")
                return output_image_path
            else:
                print("Error: Response did not contain valid JSON or image.")
                return None

    except requests.exceptions.RequestException as e:
        # Print the entire response content for more debugging information
        if e.response is not None:
            print(f"Error: {e.response.status_code} - {e.response.text}")
        else:
            print(f"Error: {str(e)}")
        return None

def generate_character(face_image_path, pose_image_path, prompt, output_image_path, api_key):
    url = "https://api.segmind.com/v1/consistent-character-with-pose"

    try:
        # Log before base64 conversion
        print("Converting face and pose images to base64.")
        face_image_b64 = to_b64(face_image_path)
        pose_image_b64 = to_b64(pose_image_path)
        print("Conversion to base64 completed.")

        # Prepare API request payload
        data = {
            "base_64": False,
            "custom_height": 1024,
            "custom_width": 1024,
            "face_image": face_image_b64,  # Convert face image to base64
            "pose_image": pose_image_b64,  # Convert pose image to base64
            "output_format": "png",
            "prompt": prompt,
            "quality": 95,
            "samples": 1,
            "seed": random.randint(0, 2**32 - 1),  # Random seed
            "use_input_img_dimension": True
        }

        headers = {'x-api-key': api_key}

        print("Sending request to character generation API.")
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # This will throw an error for HTTP issues

        # Log successful response status
        print(f"API response status: {response.status_code}")

        # Log the response content (if small enough)
        output_image_b64 = response.json().get("image")
        if output_image_b64:
            print(f"Received base64 image of length {len(output_image_b64)}")
            output_image_path = "ccgen_output.png"
            return save_base64_image(output_image_b64, output_image_path)
        else:
            print("Error: 'image' key not found in API response.")
            return None

    except requests.exceptions.JSONDecodeError:
            # If the response is not JSON, check if it's an image
            if response.headers['Content-Type'] == 'image/jpeg':
                with open(output_image_path, 'wb') as img_file:
                    img_file.write(response.content)
                print(f"Image saved directly from response: {output_image_path}")
                return output_image_path
            else:
                print("Error: Response did not contain valid JSON or image.")
                return None

    except Exception as e:
        print(f"Unexpected error in character generation: {str(e)}")
        return None

# Password handler to verify the user's password
async def handle_password(update, context):
    password = update.message.text
    correct_password = "17062004"  # Mật khẩu đúng
    username = update.message.from_user.username  # Lấy username người dùng từ tin nhắn

    if is_user_authenticated(username):  # Kiểm tra xem người dùng đã authenticated chưa
        await update.message.reply_text("You are already authenticated. You can use the inpainting and ccgen commands.")
        context.user_data['authenticated'] = True
        context.user_data['action'] = 'inpaint_input'
        return

    if password == correct_password:
        context.user_data['authenticated'] = True
        save_authenticated_user(username)  # Lưu username vào file sau khi nhập đúng mật khẩu
        await update.message.reply_text("Password correct. You are now authenticated and can proceed with inpainting.")
        context.user_data['action'] = 'inpaint_input'
    else:
        context.user_data['authenticated'] = False
        await update.message.reply_text("Incorrect password. Please try again.")
        context.user_data['action'] = 'check_password'

async def stability_inpaint(input_image_path, mask_image_path, output_path):
    # Initialize Stability API client
    stability_api = client.StabilityInference(
        key='sk-Siqk5cm2ri0QaMA1Q7ttcsrji2oLdIdh4SC1SkYZlgcvQVYZ',
        verbose=True
    )
    
    try:
        # Open and prepare the images
        init_image = Image.open(input_image_path)
        mask_image = Image.open(mask_image_path)

        # Ensure images are in RGBA mode
        if init_image.mode != 'RGBA':
            init_image = init_image.convert('RGBA')
        if mask_image.mode != 'RGBA':
            mask_image = mask_image.convert('RGBA')

        # Make API call
        answers = stability_api.generate(
            prompt="clown costume",
            init_image=init_image,
            mask_image=mask_image,
            start_schedule=1,
            seed=123456789,
            steps=30,
            cfg_scale=8.0,
            width=512,
            height=512,
            sampler=generation.SAMPLER_K_DPMPP_2M
        )

        # Process and save the result
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    img = Image.open(io.BytesIO(artifact.binary))
                    img.save(output_path)
                    return output_path

    except Exception as e:
        print(f"Error in Stability inpainting: {str(e)}")
        return None

