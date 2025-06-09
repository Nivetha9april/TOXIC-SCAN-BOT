import os
import logging
from datetime import datetime, timedelta
import psycopg2
import speech_recognition as sr
from pydub import AudioSegment
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, Filters
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from dotenv import load_dotenv
import urllib.parse as urlparse

load_dotenv()

# Logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Load tokenizer
with open('tokenizer_clean.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)
MAX_LEN = 100

# Load model
model = load_model('toxic_classifier_lstm.h5')

# PostgreSQL setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

urlparse.uses_netloc.append("postgres")
db_url = urlparse.urlparse(DATABASE_URL)
conn, cursor = None, None

def connect_db():
    global conn, cursor
    conn = psycopg2.connect(
        dbname=db_url.path[1:],
        user=db_url.username,
        password=db_url.password,
        host=db_url.hostname,
        port=db_url.port,
        sslmode='require'
    )
    cursor = conn.cursor()
    logger.info("Database connection established.")

def ensure_connection():
    global conn, cursor
    try:
        conn.poll()
    except Exception:
        logger.warning("Reconnecting DB...")
        connect_db()

connect_db()

def get_user_record(user_id):
    try:
        ensure_connection()
        cursor.execute("SELECT toxic_count, blocked_until FROM toxic_users WHERE user_id = %s", (user_id,))
        return cursor.fetchone()
    except Exception as e:
        logger.error(f"DB error: {e}")
        return None

def update_user_record(user_id, username, toxic_count, blocked_until):
    try:
        ensure_connection()
        cursor.execute("""
            INSERT INTO toxic_users (user_id, username, toxic_count, blocked_until)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET toxic_count = EXCLUDED.toxic_count,
                          blocked_until = EXCLUDED.blocked_until,
                          username = EXCLUDED.username
        """, (user_id, username, toxic_count, blocked_until))
        conn.commit()
    except Exception as e:
        logger.error(f"DB update error: {e}")

def detect_toxicity(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0][0]
    logger.info(f"Toxicity: {pred:.4f} for '{text}'")
    return 1 if pred > 0.5 else 0

TOXIC_KEYWORDS = {'hate', 'stupid', 'idiot', 'dumb', 'kill', 'trash', 'ugly'}
def explain_toxicity(text):
    return ' '.join([f"**{w}**" if w.lower() in TOXIC_KEYWORDS else w for w in text.split()])

def speech_to_text(file_path):
    wav_path = file_path.replace('.ogg', '.wav')
    try:
        sound = AudioSegment.from_file(file_path)
        sound.export(wav_path, format="wav")
        r = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
    except Exception as e:
        logger.error(f"Speech error: {e}")
        text = ""
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    return text

# Telegram Handlers
def start(update, context):
    update.message.reply_text("👋 Welcome! Send text or voice messages to check for toxicity.")

def handle_text(update, context):
    user = update.message.from_user
    user_id = str(user.id)
    username = user.username or user.first_name
    chat_id = update.message.chat_id
    message_id = update.message.message_id
    text = update.message.text
    now = datetime.now()

    record = get_user_record(user_id)
    toxic_count = 0
    blocked_until = None
    if record:
        toxic_count, blocked_until = record
        if blocked_until and now < blocked_until:
            context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            update.message.reply_text(f"⛔ Blocked until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
            return

    if detect_toxicity(text):
        toxic_count += 1
        context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        explanation = explain_toxicity(text)
        update.message.reply_text(f"🚫 Toxic message removed. Explanation:\n{explanation}")
        if toxic_count == 8:
            update.message.reply_text(f"⚠️ Warning @{username}: 8 toxic messages detected.")
        elif toxic_count >= 10:
            blocked_until = now + timedelta(days=2)
            update.message.reply_text(f"⛔ Blocked for 2 days until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        update.message.reply_text("✅ Not Toxic")

    update_user_record(user_id, username, toxic_count, blocked_until)

def handle_voice(update, context):
    user = update.message.from_user
    user_id = str(user.id)
    username = user.username or user.first_name
    chat_id = update.message.chat_id
    message_id = update.message.message_id
    now = datetime.now()

    record = get_user_record(user_id)
    toxic_count = 0
    blocked_until = None
    if record:
        toxic_count, blocked_until = record
        if blocked_until and now < blocked_until:
            context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            update.message.reply_text(f"⛔ Blocked until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
            return

    voice = update.message.voice.get_file()
    ogg_path = f"{user_id}_{message_id}.ogg"
    voice.download(ogg_path)
    context.bot.delete_message(chat_id=chat_id, message_id=message_id)

    text = speech_to_text(ogg_path)
    if os.path.exists(ogg_path):
        os.remove(ogg_path)
    if not text:
        update.message.reply_text("❓ Could not understand your voice.")
        return

    if detect_toxicity(text):
        toxic_count += 1
        explanation = explain_toxicity(text)
        update.message.reply_text(f"🚫 Toxic voice message removed. Explanation:\n{explanation}")
        if toxic_count == 8:
            update.message.reply_text(f"⚠️ Warning @{username}: 8 toxic messages detected.")
        elif toxic_count >= 10:
            blocked_until = now + timedelta(days=2)
            update.message.reply_text(f"⛔ Blocked for 2 days until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        update.message.reply_text(f"✅ Transcription: {text}\n✅ Not Toxic")

    update_user_record(user_id, username, toxic_count, blocked_until)

# Flask app
app = Flask(__name__)
TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    raise ValueError("TELEGRAM_TOKEN is not set")
bot = Bot(TOKEN)
dispatcher = Dispatcher(bot, None, workers=0, use_context=True)

# Register handlers
dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, handle_text))
dispatcher.add_handler(MessageHandler(Filters.voice, handle_voice))

@app.route("/")
def index():
    return "✅ Toxic Scan Bot is live!"

@app.route(f"/{TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

# Optional route to set webhook (call once if needed)
@app.route("/set_webhook", methods=["GET"])
def set_webhook():
    webhook_url = f"https://{request.host}/{TOKEN}"
    result = bot.set_webhook(url=webhook_url)
    return f"Webhook set: {result}"
