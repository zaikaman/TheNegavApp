from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from handlers import handle_prompt_input, handle_image, start, help_command, inpaint_command, ccgen_command, inpaint_again

# Main function to set up the bot
def main() -> None:
    telegram_api_token = "7894113437:AAGozkz4sL3L4QSwmG488rOCI923xKU3-PI"

    application = Application.builder().token(telegram_api_token).build()

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("inpaint", inpaint_command))
    application.add_handler(CommandHandler("ccgen", ccgen_command))
    application.add_handler(CommandHandler("again", inpaint_again))

    # Message handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prompt_input))
    application.add_handler(MessageHandler(filters.PHOTO, handle_image))

    application.run_polling()

if __name__ == '__main__':
    main()
