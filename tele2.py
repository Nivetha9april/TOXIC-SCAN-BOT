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

load_dotenv()

# --- Logging setup ---
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Load tokenizer ---
with open('tokenizer_clean.json', 'r', encoding='utf-8') as f:
    tokenizer_json = f.read()
tokenizer = tokenizer_from_json(tokenizer_json)

# --- Load model ---
model = load_model('toxic_classifier_lstm.h5')
MAX_LEN = 100

# --- PostgreSQL setup ---
import urllib.parse as urlparse

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set")

urlparse.uses_netloc.append("postgres")
db_url = urlparse.urlparse(DATABASE_URL)

conn = None
cursor = None

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
        logger.warning("Lost DB connection, reconnecting...")
        connect_db()

connect_db()

# --- DB functions ---
def get_user_record(user_id):
    try:
        ensure_connection()
        cursor.execute("SELECT toxic_count, blocked_until FROM toxic_users WHERE user_id = %s", (user_id,))
        return cursor.fetchone()
    except Exception as e:
        logger.error(f"DB error get_user_record: {e}")
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
        logger.error(f"DB error update_user_record: {e}")

# --- Toxicity detection ---
def detect_toxicity(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)[0][0]
    logger.info(f"Toxicity prediction for '{text}': {pred:.4f}")
    return 1 if pred > 0.5 else 0

# --- Explainability ---
TOXIC_KEYWORDS = {'hate', 'stupid', 'idiot', 'dumb', 'kill', 'trash', 'ugly'}
def explain_toxicity(text):
    words = text.split()
    highlighted = []
    for w in words:
        if w.lower() in TOXIC_KEYWORDS:
            highlighted.append(f"**{w}**")
        else:
            highlighted.append(w)
    return ' '.join(highlighted)

# --- Speech to text ---
def speech_to_text(file_path):
    wav_path = file_path.replace('.ogg', '.wav')
    try:
        sound = AudioSegment.from_file(file_path)
        sound.export(wav_path, format="wav")
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return ""

    r = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            audio = r.record(source)
        text = r.recognize_google(audio)
    except sr.UnknownValueError:
        logger.warning("Could not understand audio")
        text = ""
    except sr.RequestError as e:
        logger.error(f"Google Speech API error: {e}")
        text = ""
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
    return text

# --- Telegram handlers ---
def start(update, context):
    update.message.reply_text("👋 Welcome! Send me text or voice messages and I'll check for toxicity.")

def handle_text(update, context):
    user_id = str(update.message.from_user.id)
    username = update.message.from_user.username or update.message.from_user.first_name
    text = update.message.text
    now = datetime.now()
    chat_id = update.message.chat_id
    message_id = update.message.message_id

    record = get_user_record(user_id)
    toxic_count = 0
    blocked_until = None

    if record:
        toxic_count, blocked_until = record
        if blocked_until and now < blocked_until:
            try:
                context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception:
                pass
            update.message.reply_text(f"⛔ You're blocked until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
            return

    pred = detect_toxicity(text)

    if pred == 1:
        toxic_count += 1
        try:
            context.bot.delete_message(chat_id=chat_id, message_id=message_id)
        except Exception:
            pass
        explanation = explain_toxicity(text)
        update.message.reply_text(f"🚫 Toxic message removed. Explanation:\n{explanation}")

        if toxic_count == 8:
            update.message.reply_text(f"⚠️ Warning @{username}: 8 toxic messages detected.")
        elif toxic_count >= 10:
            blocked_until = now + timedelta(days=2)
            update.message.reply_text(f"⛔ You are blocked for 2 days until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        update.message.reply_text("✅ Not Toxic")

    update_user_record(user_id, username, toxic_count, blocked_until)

def handle_voice(update, context):
    user_id = str(update.message.from_user.id)
    username = update.message.from_user.username or update.message.from_user.first_name
    now = datetime.now()
    chat_id = update.message.chat_id
    message_id = update.message.message_id

    record = get_user_record(user_id)
    toxic_count = 0
    blocked_until = None

    if record:
        toxic_count, blocked_until = record
        if blocked_until and now < blocked_until:
            try:
                context.bot.delete_message(chat_id=chat_id, message_id=message_id)
            except Exception:
                pass
            update.message.reply_text(f"⛔ You're blocked until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
            return

    voice_file = update.message.voice.get_file()
    ogg_path = f"{user_id}_{message_id}.ogg"
    voice_file.download(ogg_path)

    text = speech_to_text(ogg_path)

    try:
        context.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except Exception:
        pass

    if os.path.exists(ogg_path):
        os.remove(ogg_path)

    if not text:
        update.message.reply_text("❓ Could not understand your voice message.")
        return

    pred = detect_toxicity(text)

    if pred == 1:
        toxic_count += 1
        explanation = explain_toxicity(text)
        update.message.reply_text(f"🚫 Toxic voice message removed. Explanation:\n{explanation}")

        if toxic_count == 8:
            update.message.reply_text(f"⚠️ Warning @{username}: 8 toxic messages detected.")
        elif toxic_count >= 10:
            blocked_until = now + timedelta(days=2)
            update.message.reply_text(f"⛔ You are blocked for 2 days until {blocked_until.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        update.message.reply_text(f"✅ Voice message transcription: {text}\n✅ Not Toxic")

    update_user_record(user_id, username, toxic_count, blocked_until)

# --- Flask app ---
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
    return "Toxic Scan Telegram Bot is running!"

@app.route("/webhook", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return "ok"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
