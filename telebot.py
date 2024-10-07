import os
import random
import requests
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import shutil
from gradio_client import Client, handle_file

# Function to generate mask using SAM API
def generate_mask(input_image_path):
    try:
        client = Client("SkalskiP/florence-sam-masking")
        result = client.predict(
            image_input=handle_file(input_image_path),
            text_input="clothing",
            api_name="/process_image"
        )

        # Print the result to inspect the response
        print(f"API Response: {result}")

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
            print("Error: Mask generation failed. Invalid result or file does not exist.")
            return None

    except Exception as e:
        print(f"Error generating mask: {str(e)}")
        return None

# Function to convert a local image to base64
def to_b64(image_path):
    with open(image_path, "rb") as image_file:
        b64_str = base64.b64encode(image_file.read()).decode('utf-8')
        print(f"Image {os.path.basename(image_path)} converted to base64, length: {len(b64_str)}")
        return b64_str

# Function to decode and save base64 image
def save_base64_image(b64_string, output_image_path):
    try:
        with open(output_image_path, "wb") as output_file:
            output_file.write(base64.b64decode(b64_string))
        print(f"Base64 string saved as image: {output_image_path}")
        return output_image_path
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return None

# Function to send request to FaceSwap API
def face_swap(source_path, target_path, output_image_path, api_key):
    url = "https://api.segmind.com/v1/faceswap-v2"

    data = {
        "source_img": to_b64(source_path),
        "target_img": to_b64(target_path),
        "input_faces_index": 0,
        "source_faces_index": 0,
        "face_restore": "codeformer-v0.1.0.pth",
        "base64": True
    }

    headers = {'x-api-key': api_key}

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        output_image_b64 = response.json().get("image")

        if output_image_b64:
            return save_base64_image(output_image_b64, output_image_path)
        else:
            print("Error: Image not found in the response.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error: {e.response.text}")
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

# Telegram bot functions
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Use /faceswap to face swap, /inpaint to start inpainting, or /again to repeat the last inpainting with new randomness.")

# /faceswap command handler
async def faceswap(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Please send the source image for face swap.")
    context.user_data['action'] = 'faceswap_source'

# Modified /inpaint command handler to start inpainting after receiving one image
async def inpaint_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Please send the input image for inpainting.")
    context.user_data['action'] = 'inpaint_input'

# /again command handler - repeats inpainting
async def inpaint_again(update: Update, context: CallbackContext) -> None:
    input_image_path = 'inpaint_input.jpg'
    mask_image_path = 'mask.jpg'
    output_image_path = 'inpaint_output.jpg'

    await update.message.reply_text("Processing the inpainting with default prompt and new randomness...")

    result = await inpaint(input_image_path, mask_image_path, output_image_path)

    if result:
        with open(output_image_path, 'rb') as img_file:
            await update.message.reply_photo(photo=img_file, caption="Here's the new inpainting result!")
    else:
        await update.message.reply_text("Failed to process the inpainting.")

# Handles incoming image messages with logic for face swap
async def handle_image(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data
    photo_file = await update.message.photo[-1].get_file()

    if user_data.get('action') == 'faceswap_source':
        # Save the source image and ask for the target image
        source_image_path = 'faceswap_source.jpg'
        await photo_file.download_to_drive(source_image_path)
        user_data['faceswap_source'] = source_image_path
        await update.message.reply_text("Source image received. Please send the target image for face swap.")
        user_data['action'] = 'faceswap_target'

    elif user_data.get('action') == 'faceswap_target':
        # Save the target image and perform face swap
        target_image_path = 'faceswap_target.jpg'
        await photo_file.download_to_drive(target_image_path)
        user_data['faceswap_target'] = target_image_path

        await update.message.reply_text("Target image received. Performing face swap...")

        output_image_path = 'faceswap_output.jpg'
        api_key = "SG_b77a34429a1aeb2e"  # Replace with your actual API key

        result = face_swap(user_data['faceswap_source'], user_data['faceswap_target'], output_image_path, api_key)

        if result:
            with open(output_image_path, 'rb') as img_file:
                await update.message.reply_photo(photo=img_file, caption="Here's the face swap result!")
        else:
            await update.message.reply_text("Failed to perform face swap.")

        # Clear user data after the process is complete
        user_data.clear()

    elif user_data.get('action') == 'inpaint_input':
        # Existing logic for inpainting
        input_image_path = 'inpaint_input.jpg'
        await photo_file.download_to_drive(input_image_path)
        await update.message.reply_text("Input image received. Generating mask...")

        mask_image_path = generate_mask(input_image_path)

        if mask_image_path:
            await update.message.reply_text("Mask generated successfully. Processing inpainting...")

            output_image_path = 'inpaint_output.jpg'
            result = await inpaint(input_image_path, mask_image_path, output_image_path)

            if result:
                with open(output_image_path, 'rb') as img_file:
                    await update.message.reply_photo(photo=img_file, caption="Here's the inpaint result!")
            else:
                await update.message.reply_text("Failed to process the inpainting.")
        else:
            await update.message.reply_text("Failed to generate mask.")

        user_data.clear()

# Main function to set up the bot
def main() -> None:
    telegram_api_token = "7894113437:AAGozkz4sL3L4QSwmG488rOCI923xKU3-PI"

    application = Application.builder().token(telegram_api_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("faceswap", faceswap))
    application.add_handler(CommandHandler("inpaint", inpaint_command))
    application.add_handler(CommandHandler("again", inpaint_again))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.run_polling()

if __name__ == '__main__':
    main()