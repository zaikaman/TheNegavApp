import random
import os
import shutil
import base64
import requests
from telegram import Update
from telegram.ext import CallbackContext
from bot_utils import to_b64, save_base64_image, generate_mask, inpaint, face_swap, generate_character, handle_password


# Telegram bot functions
async def start(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Welcome to the Negav bot, use /faceswap to face swap, /inpaint to start inpainting, /again to repeat the last inpainting with new randomness, or /help to show all available commands.")

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

# /faceswap command handler
async def faceswap(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Please send the source image for face swap.")
    context.user_data['action'] = 'faceswap_source'

# Modified /inpaint command handler to ask for a password before inpainting
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

# /ccgen command handler (character generation)
async def ccgen_command(update: Update, context: CallbackContext) -> None:
    await update.message.reply_text("Please send the face image for character generation.")
    context.user_data['action'] = 'ccgen_face'

# Handle user text input (for prompt or password)
async def handle_prompt_input(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data

    # Handle character generation prompt input
    if user_data.get('action') == 'ccgen_prompt':
        prompt = update.message.text
        user_data['ccgen_prompt'] = prompt
        await update.message.reply_text(f"Prompt received: '{prompt}'. Generating the character...")

        # Perform character generation after receiving face, pose, and prompt
        face_image_path = user_data['ccgen_face']
        pose_image_path = user_data['ccgen_pose']
        output_image_path = 'ccgen_output.jpg'
        api_key = "SG_03c17009344e9a6c"
        result = generate_character(face_image_path, pose_image_path, prompt, output_image_path, api_key)

        if result:
            with open(output_image_path, 'rb') as img_file:
                await update.message.reply_photo(photo=img_file, caption="Here's the generated character!")
        else:
            await update.message.reply_text("Failed to generate the character.")
        user_data.clear()

    # Handle password input (no changes)
    elif user_data.get('action') == 'check_password':
        await handle_password(update, context)

# Handle images for face swap, inpainting, and character generation
async def handle_image(update: Update, context: CallbackContext) -> None:
    user_data = context.user_data
    photo_file = await update.message.photo[-1].get_file()

    # Face swap: Handle source image
    if user_data.get('action') == 'faceswap_source':
        source_image_path = 'faceswap_source.jpg'
        await photo_file.download_to_drive(source_image_path)
        user_data['faceswap_source'] = source_image_path
        await update.message.reply_text("Source image received. Please send the target image for face swap.")
        user_data['action'] = 'faceswap_target'

    # Face swap: Handle target image
    elif user_data.get('action') == 'faceswap_target':
        target_image_path = 'faceswap_target.jpg'
        await photo_file.download_to_drive(target_image_path)
        user_data['faceswap_target'] = target_image_path
        await update.message.reply_text("Target image received. Performing face swap...")

        output_image_path = 'faceswap_output.jpg'
        api_key = "SG_b77a34429a1aeb2e"
        result = face_swap(user_data['faceswap_source'], user_data['faceswap_target'], output_image_path, api_key)

        if result:
            with open(output_image_path, 'rb') as img_file:
                await update.message.reply_photo(photo=img_file, caption="Here's the face swap result!")
        else:
            await update.message.reply_text("Failed to perform face swap.")
        user_data.clear()

    # Inpainting: Handle input image
    elif user_data.get('action') == 'inpaint_input':
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
