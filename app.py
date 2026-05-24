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

app = Flask(__name__)

# ---------------------------------------------------
# Globals
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
    le         = joblib.load('label_encoder.pkl')
    CLASS_NAMES = list(le.classes_)
    print("Model files loaded successfully.")
    print("Classes:", CLASS_NAMES)

except Exception as e:
    print("Error loading model files:", str(e))
    le          = None
    CLASS_NAMES = []


# ---------------------------------------------------
# Text Cleaning Functions
# ---------------------------------------------------

def strip_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace='')


def expand_contractions_fn(text: str) -> str:
    return contractions.fix(text)


def strip_all_entities(text: str) -> str:
    """Lowercase, remove links/mentions, non-ASCII, punctuation, stopwords."""
    text = re.sub(r'\r|\n', ' ', text.lower())
    text = re.sub(r'(?:\@|https?://|www\.)\S+', '', text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    table = str.maketrans('', '', string.punctuation)
    text = text.translate(table)
    return ' '.join(w for w in text.split() if w not in STOP_WORDS)


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
    Inference cleaning pipeline — matches v2 notebook exactly,
    with one intentional difference:

    filter_non_english() is REMOVED from the inference path.

    Reason: langdetect is unreliable on short texts (1-3 words)
    that remain after stopword removal. For example:
      "You are a liar"  -->  stopwords strip to  "liar"
      langdetect("liar") may return non-English --> empty string --> 422 error.

    Language filtering was a DATA CLEANING step for training only
    (to remove foreign-language tweets from the dataset).
    At inference time we predict on whatever text is given;
    non-English input will produce a low-confidence prediction,
    which the caller can handle via the confidence score.
    """
    tweet = strip_emoji(tweet)
    tweet = expand_contractions_fn(tweet)
    tweet = strip_all_entities(tweet)
    # filter_non_english intentionally omitted at inference time
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
        "status":       "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "classes":      CLASS_NAMES
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

        # If cleaning produces an empty string (e.g. input was only
        # punctuation / numbers / emoji), fall back to a minimal clean
        # so the vectorizer always receives something.
        if not cleaned.strip():
            fallback = re.sub(r'\s+', ' ', text.lower().strip())
            fallback = fallback.translate(
                str.maketrans('', '', string.punctuation)
            ).strip()
            cleaned = fallback if fallback.strip() else text.lower().strip()

        # ── Vectorise ────────────────────────────────
        cv_vector    = vectorizer.transform([cleaned])
        tfidf_vector = tfidf.transform(cv_vector)

        # ── Predict ──────────────────────────────────
        pred_label = model.predict(tfidf_vector)[0]
        confidence = float(model.predict_proba(tfidf_vector)[0].max())

        # Decode via LabelEncoder — never a hardcoded dict
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
