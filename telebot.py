import os
import random
import requests
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
import shutil
from gradio_client import Client, handle_file

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
    
# /ccgen command handler to start the character generation process
async def ccgen_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Please send the face image for character generation.")
    context.user_data['action'] = 'ccgen_face'
    
# /ccgenprompt command handler to ask for the custom prompt
async def ccgenprompt_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Enter your prompt:")
    context.user_data['action'] = 'ccgenprompt_waiting'  # Set the action to wait for the prompt
    print("Waiting for user prompt...")  # Add a log to track this step
    
# Function to handle user text input for the prompt
async def handle_prompt_input(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data

    # Check if the bot is waiting for a prompt
    if user_data.get('action') == 'ccgenprompt_waiting':
        prompt = update.message.text
        user_data['ccgen_prompt'] = prompt  # Store the prompt in user_data
        await update.message.reply_text(f"Prompt received: '{prompt}'. Now, you can continue with the character generation.")

        # Optionally, clear the action after receiving the prompt
        user_data['action'] = None
    
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
    await update.message.reply_text("Welcome to the Negav bot, use /faceswap to face swap, /inpaint to start inpainting, /again to repeat the last inpainting with new randomness, or /help to show all available commands.")

# /faceswap command handler
async def faceswap(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Please send the source image for face swap.")
    context.user_data['action'] = 'faceswap_source'

# Modified /inpaint command handler to ask for a password before inpainting
async def inpaint_command(update: Update, context: CallbackContext) -> None:
    # Check if the user is already authenticated
    if context.user_data.get('authenticated', False):
        await update.message.reply_text("Please send the input image for inpainting.")
        context.user_data['action'] = 'inpaint_input'
    else:
        # Ask for the password if not authenticated
        await update.message.reply_text("What's the password?")
        context.user_data['action'] = 'check_password'
        
# Function to handle user input and check password
async def handle_password(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data

    if user_data.get('action') == 'check_password':
        password = update.message.text

        if password == "17062004":
            await update.message.reply_text("Password correct! You can now use the inpaint feature.")
            context.user_data['authenticated'] = True
            await inpaint_command(update, context)  # Proceed to the inpaint command
        else:
            await update.message.reply_text("Wrong password. You can't use this feature right now.")
            user_data['authenticated'] = False
    
# New /help command handler
async def help_command(update: Update, context: CallbackContext) -> None:
    help_text = (
        "Available commands:\n"
        "/start - Welcome message\n"
        "/help - Show this help message\n"
        "/faceswap - Start the face swap process\n"
        "/inpaint - Start the inpainting process\n"
        "/again - Repeat the last inpainting with new randomness."
    )
    await update.message.reply_text(help_text)

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

async def handle_image(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data
    photo_file = await update.message.photo[-1].get_file()

    # Inpainting logic
    if user_data.get('action') == 'inpaint_input':
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

    # ccgen face image handling
    if user_data.get('action') == 'ccgen_face':
        face_image_path = 'ccgen_face.jpg'
        await photo_file.download_to_drive(face_image_path)
        user_data['ccgen_face'] = face_image_path
        await update.message.reply_text("Face image received. Please send the pose image for character generation.")
        user_data['action'] = 'ccgen_pose'

    # ccgen pose image handling
    elif user_data.get('action') == 'ccgen_pose':
        pose_image_path = 'ccgen_pose.jpg'
        await photo_file.download_to_drive(pose_image_path)
        user_data['ccgen_pose'] = pose_image_path

        # Now use the prompt that was set via /ccgenprompt
        prompt = user_data.get('ccgen_prompt', 'naked, perfect body')

        await update.message.reply_text(f"Pose image received. Generating the character with the prompt: '{prompt}'...")

        # Perform character generation
        output_image_path = 'ccgen_output.jpg'
        api_key = "SG_b77a34429a1aeb2e"
        result = generate_character(user_data['ccgen_face'], user_data['ccgen_pose'], prompt, output_image_path, api_key)

        if result:
            with open(output_image_path, 'rb') as img_file:
                await update.message.reply_photo(photo=img_file, caption="Here's the generated character!")
        else:
            await update.message.reply_text("Failed to generate the character.")

        user_data.clear()  # Clear user data after character generation

    # If the bot is waiting for a password to proceed
    elif user_data.get('action') == 'check_password':
        await handle_password(update, context)

# Main function to set up the bot
def main() -> None:
    telegram_api_token = "7894113437:AAGozkz4sL3L4QSwmG488rOCI923xKU3-PI"

    application = Application.builder().token(telegram_api_token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))  # Add help command handler
    application.add_handler(CommandHandler("faceswap", faceswap))
    application.add_handler(CommandHandler("inpaint", inpaint_command))
    application.add_handler(CommandHandler("ccgen", ccgen_command))  # Add /ccgen command handler
    application.add_handler(CommandHandler("ccgenprompt", ccgenprompt_command))  # New prompt command
    application.add_handler(CommandHandler("again", inpaint_again))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prompt_input))  # Handle password input
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))  # Handle image uploads
    application.run_polling()

if __name__ == '__main__':
    main()