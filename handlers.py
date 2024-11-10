import random
import os
import shutil
import base64
import requests
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext
from bot_utils import to_b64, save_base64_image, generate_mask, inpaint, generate_character, handle_password, stability_inpaint


# Telegram bot functions
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Welcome to the Negav bot, use /inpaint to start inpainting, /again to repeat the last inpainting with new randomness, /ccgen to generate consistent character or /help to show all available commands.")

# New /help command handler
async def help_command(update: Update, context: CallbackContext) -> None:
    help_text = (
        "Available commands:\n"
        "/start - Welcome message\n"
        "/help - Show this help message\n"
        "/inpaint - Start the inpainting process\n"
        "/again - Repeat the last inpainting with new randomness."
    )
    await update.message.reply_text(help_text)

# Modified /inpaint command handler to ask for a password before inpainting
async def inpaint_command(update: Update, context: CallbackContext) -> None:
    username = update.message.from_user.username
    
    if is_user_authenticated(username):
        keyboard = [
            [InlineKeyboardButton("Segmind", callback_data='api_segmind'),
             InlineKeyboardButton("Stability AI", callback_data='api_stability')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Please choose which API to use:",
            reply_markup=reply_markup
        )
    else:
        await update.message.reply_text("Please enter the password to continue.")
        context.user_data['action'] = 'check_password'

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

# /ccgen command handler (character generation)
async def ccgen_command(update: Update, context: CallbackContext) -> None:
    username = update.message.from_user.username
    
    # Kiểm tra xem user đã authenticated chưa
    if is_user_authenticated(username):
        await update.message.reply_text("You are authenticated. Please send the face image for character generation.")
        context.user_data['action'] = 'ccgen_face'
    else:
        await update.message.reply_text("Please enter the password to continue.")
        context.user_data['action'] = 'check_password'

async def handle_prompt_input(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data
    username = update.message.from_user.username

    # Nếu user đã authenticated thì bỏ qua việc yêu cầu nhập mật khẩu
    if is_user_authenticated(username):
        user_data['authenticated'] = True

    # Handle character generation prompt input
    if user_data.get('action') == 'ccgen_prompt':
        prompt = update.message.text
        user_data['ccgen_prompt'] = prompt
        await update.message.reply_text(f"Prompt received: '{prompt}'. Generating the character...")

        # Perform character generation after receiving face, pose, and prompt
        face_image_path = user_data['ccgen_face']
        pose_image_path = user_data['ccgen_pose']
        output_image_path = 'ccgen_output.jpg'
        api_key = "SG_38bb0c4a24258d71"
        result = generate_character(face_image_path, pose_image_path, prompt, output_image_path, api_key)

        if result:
            with open(output_image_path, 'rb') as img_file:
                await update.message.reply_photo(photo=img_file, caption="Here's the generated character!")
        else:
            await update.message.reply_text("Failed to generate the character.")
        user_data.clear()

    # Handle password input (for users not authenticated)
    elif not user_data.get('authenticated'):
        await handle_password(update, context)

# Handle images for face swap, inpainting, and character generation
async def handle_image(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data
    username = update.message.from_user.username

    # Nếu người dùng chưa authenticated, yêu cầu nhập mật khẩu
    if not is_user_authenticated(username) and not user_data.get('authenticated'):
        await update.message.reply_text("Please enter the password to proceed.")
        context.user_data['action'] = 'check_password'
        return

    photo_file = await update.message.photo[-1].get_file()

    # Inpainting: Handle input image
    if user_data.get('action') == 'inpaint_input':
        input_image_path = 'inpaint_input.jpg'
        await photo_file.download_to_drive(input_image_path)
        await update.message.reply_text("Input image received. Generating mask...")

        mask_image_path = generate_mask(input_image_path)

        if mask_image_path:
            await update.message.reply_text("Mask generated successfully. Processing inpainting...")
            output_image_path = 'inpaint_output.jpg'
            
            # Choose API based on user selection
            if user_data.get('api_choice') == 'stability':
                result = await stability_inpaint(input_image_path, mask_image_path, output_image_path)
            else:  # default to segmind
                result = await inpaint(input_image_path, mask_image_path, output_image_path)

            if result:
                with open(output_image_path, 'rb') as img_file:
                    await update.message.reply_photo(photo=img_file)
            else:
                await update.message.reply_text("Failed to process the inpainting.")
        else:
            await update.message.reply_text("Failed to generate mask.")
        user_data.clear()

    # Character generation: Handle face image
    elif user_data.get('action') == 'ccgen_face':
        face_image_path = 'ccgen_face.jpg'
        await photo_file.download_to_drive(face_image_path)
        user_data['ccgen_face'] = face_image_path
        await update.message.reply_text("Face image received. Please send the pose image for character generation.")
        user_data['action'] = 'ccgen_pose'

    # Character generation: Handle pose image
    elif user_data.get('action') == 'ccgen_pose':
        pose_image_path = 'ccgen_pose.jpg'
        await photo_file.download_to_drive(pose_image_path)
        user_data['ccgen_pose'] = pose_image_path

        # Now wait for the user to send the prompt
        await update.message.reply_text("Pose image received. Please type the prompt for character generation.")
        user_data['action'] = 'ccgen_prompt'  # Set the action to expect a prompt input
        
async def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()
    
    if query.data == 'api_segmind':
        context.user_data['api_choice'] = 'segmind'
    elif query.data == 'api_stability':
        context.user_data['api_choice'] = 'stability'
    
    await query.edit_message_text("Please send the input image for inpainting.")
    context.user_data['action'] = 'inpaint_input'

def is_user_authenticated(username: str) -> bool:
    try:
        with open('authenticated_users.txt', 'r') as f:
            authenticated_users = f.read().splitlines()
        return username in authenticated_users
    except FileNotFoundError:
        return False