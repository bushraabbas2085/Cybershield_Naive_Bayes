from flask import Flask, request, jsonify
import joblib
import re
import string
import emoji
import contractions
import traceback
import os

import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4',  quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect, LangDetectException

app = Flask(__name__)

# ---------------------------------------------------
# Globals — identical to v2 notebook
# ---------------------------------------------------
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

# ---------------------------------------------------
# Load Saved Model Files
# ---------------------------------------------------
try:
    model      = joblib.load('naive_bayes_model.pkl')
    vectorizer = joblib.load('count_vectorizer.pkl')
    tfidf      = joblib.load('tfidf_transformer.pkl')
    le         = joblib.load('label_encoder.pkl')   # FIX: use LabelEncoder, not hardcoded dict
    CLASS_NAMES = list(le.classes_)
    print("Model files loaded successfully.")
    print("Classes:", CLASS_NAMES)

except Exception as e:
    print("Error loading model files:", str(e))
    le          = None
    CLASS_NAMES = []

# ---------------------------------------------------
# Text Cleaning — exact copy of v2 notebook pipeline
# ---------------------------------------------------

def strip_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace='')


def expand_contractions_fn(text: str) -> str:
    return contractions.fix(text)


def strip_all_entities(text: str) -> str:
    """Lowercase, remove links / mentions, non-ASCII, punctuation, stopwords."""
    text = re.sub(r'\r|\n', ' ', text.lower())
    text = re.sub(r'(?:\@|https?://|www\.)\S+', '', text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    return ' '.join(w for w in text.split() if w not in STOP_WORDS)


def filter_non_english(text: str) -> str:
    """
    Return empty string for non-English text.
    Called AFTER strip_all_entities so raw URLs / mentions
    do not confuse langdetect.
    """
    if not text.strip():
        return ''
    try:
        lang = detect(text)
    except LangDetectException:
        lang = 'unknown'
    return text if lang == 'en' else ''


def clean_hashtags(tweet: str) -> str:
    tweet = re.sub(r'(\s+#[\w-]+)+\s*$', '', tweet).strip()
    return re.sub(r'#([\w-]+)', r'\1', tweet).strip()


def filter_chars(text: str) -> str:
    return ' '.join('' if ('$' in w or '&' in w) else w for w in text.split())


def remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)


def lemmatize_text(text: str) -> str:
    return ' '.join(LEMMATIZER.lemmatize(w) for w in word_tokenize(text))


def remove_short_words(text: str, min_len: int = 2) -> str:
    return ' '.join(w for w in text.split() if len(w) >= min_len)


def replace_elongated(text: str) -> str:
    return re.sub(r'\b(\w+)((\w)\3{2,})(\w*)\b', r'\1\3\4', text)


def remove_url_shorteners(text: str) -> str:
    return re.sub(
        r'(?:http[s]?://)?(?:www\.)?'
        r'(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|ow\.ly|bit\.do)\S+',
        '', text
    )


def clean_tweet(tweet: str) -> str:
    """
    Full v2 cleaning pipeline.
    Order matters: strip_all_entities runs BEFORE filter_non_english
    so noise (URLs, @mentions) does not confuse the language detector.
    """
    tweet = strip_emoji(tweet)
    tweet = expand_contractions_fn(tweet)
    tweet = strip_all_entities(tweet)       # clean first …
    tweet = filter_non_english(tweet)       # … then detect language
    tweet = clean_hashtags(tweet)
    tweet = filter_chars(tweet)
    tweet = remove_numbers(tweet)
    tweet = lemmatize_text(tweet)
    tweet = remove_short_words(tweet)
    tweet = replace_elongated(tweet)
    tweet = remove_url_shorteners(tweet)
    tweet = ' '.join(tweet.split())
    return tweet.strip()

# ---------------------------------------------------
# Home Route
# ---------------------------------------------------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Cyberbullying Detection API Running Successfully",
        "classes": CLASS_NAMES
    })

# ---------------------------------------------------
# Health Check Route
# ---------------------------------------------------
@app.route('/health', methods=['GET'])
def health():
    model_loaded = le is not None
    return jsonify({
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "classes": CLASS_NAMES
    }), 200 if model_loaded else 503

# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:

        # ── Guard: model not loaded ──────────────────
        if le is None:
            return jsonify({
                "success": False,
                "error": "Model files are not loaded. Check server logs."
            }), 503

        # ── Validate request ─────────────────────────
        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data received"
            }), 400

        if 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'text' field"
            }), 400

        text = data['text']

        if not isinstance(text, str) or text.strip() == "":
            return jsonify({
                "success": False,
                "error": "Input text cannot be empty"
            }), 400

        # ── Clean ────────────────────────────────────
        cleaned = clean_tweet(text)

        # If cleaning wipes the text entirely (e.g. non-English tweet),
        # return a graceful response instead of predicting on an empty string.
        if not cleaned.strip():
            return jsonify({
                "success": False,
                "error": "Text is empty after cleaning (possibly non-English or only special characters)."
            }), 422

        # ── Vectorise ────────────────────────────────
        cv_vector    = vectorizer.transform([cleaned])
        tfidf_vector = tfidf.transform(cv_vector)

        # ── Predict ──────────────────────────────────
        pred_label = model.predict(tfidf_vector)[0]
        confidence = float(model.predict_proba(tfidf_vector)[0].max())

        # FIX: decode via LabelEncoder — not a hardcoded dict.
        # This always matches whichever classes were present during training.
        result = le.inverse_transform([pred_label])[0]

        # ── Response ─────────────────────────────────
        return jsonify({
            "success":      True,
            "input_text":   text,
            "cleaned_text": cleaned,
            "prediction":   result,
            "confidence":   round(confidence, 4)
        })

    except Exception as e:
        print("Prediction Error")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ---------------------------------------------------
# Handle Invalid Routes
# ---------------------------------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "success": False,
        "error": "Route not found"
    }), 404

# ---------------------------------------------------
# Handle Internal Server Errors
# ---------------------------------------------------
@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# ---------------------------------------------------
# Run Flask App
# ---------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)