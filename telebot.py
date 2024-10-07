import os
import requests
import base64
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from io import BytesIO

# Function to convert a local image to base64
def to_b64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to send request to FaceSwap API
def face_swap(source_path, target_path, output_image_path, api_key):
    url = "https://api.segmind.com/v1/faceswap-v2"
    
    # Prepare the request data
    data = {
        "source_img": to_b64(source_path),
        "target_img": to_b64(target_path),
        "input_faces_index": 0,
        "source_faces_index": 0,
        "face_restore": "codeformer-v0.1.0.pth",
        "base64": True
    }
    
    # API request headers
    headers = {
        'x-api-key': api_key
    }
    
    try:
        # Send the POST request to the API
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses

        # Extract the base64 image from the response
        output_image_b64 = response.json().get("image")
        
        if output_image_b64:
            # Decode and save the base64-encoded image to output_image_path
            with open(output_image_path, "wb") as output_file:
                output_file.write(base64.b64decode(output_image_b64))
            return output_image_path
        else:
            return None
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e.response.text}")
        return None

# Telegram bot functions
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Send me two images: a source and a target, and I will swap the faces!")

async def receive_image(update: Update, context: CallbackContext) -> None:
    user_images = context.user_data.setdefault("images", [])
    
    # Get the image from the message
    photo_file = await update.message.photo[-1].get_file()
    
    # Save the image in the directory with the correct name
    if len(user_images) == 0:
        await photo_file.download_to_drive('zan.jpg')  # Save first image as zan.jpg
        await update.message.reply_text("Received source image! Now send the target image.")
        user_images.append('zan.jpg')
    elif len(user_images) == 1:
        await photo_file.download_to_drive('target.jpg')  # Save second image as target.jpg
        await update.message.reply_text("Received target image! Processing face swap...")
        user_images.append('target.jpg')

        # Perform face swap
        output_image_path = 'output.jpg'
        api_key = "SG_92402b81cb67df50"  # Your API Key
        
        result = face_swap('zan.jpg', 'target.jpg', output_image_path, api_key)
        
        if result:
            # Send the output image back to the user
            with open(output_image_path, 'rb') as img_file:
                await update.message.reply_photo(photo=img_file, caption="Here's the result!")
        else:
            await update.message.reply_text("Failed to process the image. Please try again.")

        # Clear user data after processing
        context.user_data["images"] = []
    else:
        await update.message.reply_text("You have already submitted two images. Please wait while I process them.")

def main() -> None:
    # Your bot's API Token
    telegram_api_token = "7894113437:AAGozkz4sL3L4QSwmG488rOCI923xKU3-PI"
    
    # Create an Application object (formerly Updater)
    application = Application.builder().token(telegram_api_token).build()
    
    # Command handler for '/start'
    application.add_handler(CommandHandler("start", start))
    
    # Message handler for receiving images
    application.add_handler(MessageHandler(filters.PHOTO, receive_image))
    
    # Start the bot
    application.run_polling()

if __name__ == '__main__':
    main()
