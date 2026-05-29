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
# Discriminatory keyword patterns — one per class
# (mirrors the patterns used in v2 notebook training)
# ---------------------------------------------------
_CLASS_PATTERNS = {
    'religion': re.compile(
        r'\b(god|allah|muslim|islam\w*|christian\w*|jew\w*|judais\w*|hindu\w*|'
        r'religion\w*|religious|church|mosque|temple|bible|quran|faith|pray\w*|'
        r'prophet|atheis\w*|pagan|cult|sect|belief|spiritual\w*|deity|divine|'
        r'sacred|holy|blasphemy|apostate|kafir|infidel|heathen|sikh\w*|buddhis\w*|'
        r'pope|priest|imam|rabbi|pastor|theology|jihad|evangelical|catholic|'
        r'protestant|orthodox|sharia|halal|haram|scripture|salvation|worship|'
        r'sermon|messiah|savior|covenant)\b',
        re.IGNORECASE
    ),
    'ethnicity': re.compile(
        r'\b(black|white|asian\w*|hispanic\w*|latino\w*|latina\w*|arab\w*|'
        r'indian\w*|chinese|african\w*|mexican\w*|race|racial|racist\w*|racism|'
        r'ethnic\w*|minorit\w*|immigrant\w*|immigration|nigger|nigga|cracker|'
        r'spic|chink|foreigner\w*|native\w*|caucasian|negro|kkk|apartheid|'
        r'segregation|stereotype\w*|prejudice\w*|bigot\w*|xenophob\w*|'
        r'nationalit\w*|deportation)\b',
        re.IGNORECASE
    ),
    'gender': re.compile(
        r'\b(woman|women|man\b|men\b|girl\w*|boy\w*|female|male|gender\w*|'
        r'sexis\w*|feminist\w*|feminism|gay|lesbian|trans\w*|lgbt\w*|queer|'
        r'bisexual|homosexual|heterosexual|rape\w*|sexual\w*|harassment|'
        r'misogyn\w*|patriarchy|matriarchy|nonbinary|intersex|pronoun\w*|'
        r'abortion|mansplain\w*|incel\w*|simp|thot|kitchen|masculine|feminine)\b',
        re.IGNORECASE
    ),
    'age': re.compile(
        r'\b(old\b|young\b|age\b|aged|teen\w*|adult\w*|kid\b|kids\b|child\w*|'
        r'elder\w*|senior\w*|minor\b|youth|boomer\w*|millennial\w*|zoomer\w*|'
        r'generation|retire\w*|ancient|immature|underage|juvenile|toddler|'
        r'infant|baby|grann\w*|grandma|grandpa|pensioner|geriatric|wrinkle\w*|'
        r'fossil|ageism|ageist|too\s+old|too\s+young|grow\s+up|act\s+your\s+age)\b',
        re.IGNORECASE
    ),
}

# Offensive / harmful language that signals cyberbullying even without a
# specific class keyword — lets the model decide the class in these cases.
_OFFENSIVE_PATTERN = re.compile(
    r'\b(fuck\w*|bull\s*shit|shit\w*|bitch\w*|ass\s*hole|bastard|whore|slut|'
    r'cunt|nigger|nigga|faggot|retard\w*|dumb\s*ass|moron|kill\s+\w*self|'
    r'go\s+fuck|fuck\s+(?:you|off)|motherfuck\w*|piece\s+of\s+shit|'
    r'son\s+of\s+a\s+bitch|go\s+to\s+hell|drop\s+dead|worthless|'
    r'i\s+hate\s+you|you\s+should\s+die|go\s+kill|stupid\s+idiot|'
    r'disgusting|horrible\s+person|you\s+deserve\s+to|ugly\s+\w+|'
    r'loser\b|freak\b|trash\b|scum\b|pathetic\b)\b',
    re.IGNORECASE
)


def _keyword_override(text: str):
    """
    Rule-based pre-check BEFORE the model.

    Returns 'not_cyberbullying' with confidence 1.0 when the text contains
    NO discriminatory class keywords AND NO offensive language.

    This fixes the case where neutral/positive tweets ('You are a good person',
    'Happy birthday', 'You are amazing') contain no harmful signal at all —
    the model alone cannot reliably distinguish these from cyberbullying tweets
    because the training data's not_cyberbullying class is noisy.

    Returns None when the text should be passed to the model normally.
    """
    for pat in _CLASS_PATTERNS.values():
        if pat.search(text):
            return None                  # has discriminatory keyword → model decides

    if _OFFENSIVE_PATTERN.search(text):
        return None                      # has offensive language → model decides

    return 'not_cyberbullying'           # no harmful signal → safe override


# ---------------------------------------------------
# Load Saved Model Files
# ---------------------------------------------------
try:
    model       = joblib.load('naive_bayes_model.pkl')
    vectorizer  = joblib.load('count_vectorizer.pkl')
    tfidf       = joblib.load('tfidf_transformer.pkl')
    le          = joblib.load('label_encoder.pkl')
    CLASS_NAMES = list(le.classes_)
    print("Model files loaded successfully.")
    print("Classes:", CLASS_NAMES)

except Exception as e:
    print("Error loading model files:", str(e))
    le          = None
    CLASS_NAMES = []


# ---------------------------------------------------
# Text Cleaning Functions (exact v2 notebook pipeline)
# ---------------------------------------------------

def _strip_emoji(text: str) -> str:
    return emoji.replace_emoji(text, replace='')


def _expand_contractions(text: str) -> str:
    return contractions.fix(text)


def _strip_all_entities(text: str) -> str:
    """Lowercase, remove links/mentions, non-ASCII, punctuation, stopwords."""
    text = re.sub(r'\r|\n', ' ', text.lower())
    text = re.sub(r'(?:\@|https?://|www\.)\S+', '', text)
    text = re.sub(r'[^\x00-\x7f]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(w for w in text.split() if w not in STOP_WORDS)


def _clean_hashtags(text: str) -> str:
    text = re.sub(r'(\s+#[\w-]+)+\s*$', '', text).strip()
    return re.sub(r'#([\w-]+)', r'\1', text).strip()


def _filter_chars(text: str) -> str:
    return ' '.join('' if ('$' in w or '&' in w) else w for w in text.split())


def _remove_numbers(text: str) -> str:
    return re.sub(r'\d+', '', text)


def _lemmatize(text: str) -> str:
    return ' '.join(LEMMATIZER.lemmatize(w) for w in word_tokenize(text))


def _remove_short_words(text: str, min_len: int = 2) -> str:
    return ' '.join(w for w in text.split() if len(w) >= min_len)


def _replace_elongated(text: str) -> str:
    return re.sub(r'\b(\w+)((\w)\3{2,})(\w*)\b', r'\1\3\4', text)


def _remove_url_shorteners(text: str) -> str:
    return re.sub(
        r'(?:http[s]?://)?(?:www\.)?'
        r'(?:bit\.ly|goo\.gl|t\.co|tinyurl\.com|ow\.ly|bit\.do)\S+',
        '', text
    )


def clean_tweet(tweet: str) -> str:
    """
    Full v2 cleaning pipeline.

    NOTE: filter_non_english() is intentionally omitted at inference time.
    langdetect is unreliable on short texts (1–3 words remaining after
    stopword removal) and would silently empty valid short English tweets.
    Language filtering is a training-data step only.
    """
    tweet = _strip_emoji(tweet)
    tweet = _expand_contractions(tweet)
    tweet = _strip_all_entities(tweet)
    tweet = _clean_hashtags(tweet)
    tweet = _filter_chars(tweet)
    tweet = _remove_numbers(tweet)
    tweet = _lemmatize(tweet)
    tweet = _remove_short_words(tweet)
    tweet = _replace_elongated(tweet)
    tweet = _remove_url_shorteners(tweet)
    return ' '.join(tweet.split()).strip()


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

        # ── Guard: model not loaded ─────────────────
        if le is None:
            return jsonify({
                "success": False,
                "error": "Model files are not loaded. Check server logs."
            }), 503

        # ── Validate request ────────────────────────
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

        if not isinstance(text, str) or not text.strip():
            return jsonify({
                "success": False,
                "error": "Input text cannot be empty"
            }), 400

        # ── Rule-based override (before model) ──────
        # If the raw text contains no discriminatory keyword and no offensive
        # language, return not_cyberbullying immediately with confidence 1.0.
        # This reliably handles neutral/positive tweets that the noisy training
        # data causes the model to misclassify.
        override = _keyword_override(text)
        if override:
            return jsonify({
                "success":      True,
                "input_text":   text,
                "cleaned_text": text,
                "prediction":   override,
                "confidence":   1.0,
                "method":       "keyword_override"
            })

        # ── Clean ───────────────────────────────────
        cleaned = clean_tweet(text)

        # Fallback: if cleaning empties the string, use a minimal clean
        # so the vectorizer always receives a non-empty input.
        if not cleaned.strip():
            cleaned = text.lower().translate(
                str.maketrans('', '', string.punctuation)
            ).strip()

        # ── Vectorise ───────────────────────────────
        cv_vector    = vectorizer.transform([cleaned])
        tfidf_vector = tfidf.transform(cv_vector)

        # ── Predict ─────────────────────────────────
        pred_label = model.predict(tfidf_vector)[0]
        confidence = float(model.predict_proba(tfidf_vector)[0].max())

        # Decode via LabelEncoder — never a hardcoded dict
        result = le.inverse_transform([pred_label])[0]

        # ── Response ────────────────────────────────
        return jsonify({
            "success":      True,
            "input_text":   text,
            "cleaned_text": cleaned,
            "prediction":   result,
            "confidence":   round(confidence, 4),
            "method":       "model"
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
