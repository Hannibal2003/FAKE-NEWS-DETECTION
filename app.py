from flask import Flask, render_template, request
import joblib
import google.generativeai as genai
import re
import string

# Load ML model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Flask app
app = Flask(__name__)

# Gemini config
genai.configure(api_key="AIzaSyAt-GvdPJOaK_0JJ9-MZtOWzNe3LeRGSVc")
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Simple text cleaning (same as wordopt)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Analyze route
@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text')
    is_url = text.startswith(('http://', 'https://'))

    # --- ML model prediction ---
    cleaned_text = clean_text(text)
    xv = vectorizer.transform([cleaned_text])
    prediction = model.predict(xv)[0]
    ml_result = "Fake News" if prediction == 0 else "Not Fake News"

    # --- Gemini Analysis ---
    try:
        verification_prompt = f"""Analyze this {'news article' if is_url else 'text'} for veracity:
        {text}
        
        Provide:
        1. Source credibility assessment
        2. Cross-referenced facts from recent news (up to now)
        3. Potential biases detected
        4. Likelihood score (0-100%) of being misinformation
        5. Recommended fact-checking sources with URLs
        6. Relevant quotes from authoritative sources"""

        response = gemini_model.generate_content(verification_prompt)
        gemini_result = response.text
    except Exception as e:
        gemini_result = f"Error analyzing content: {str(e)}"

    return render_template('index.html', result=gemini_result, ml_result=ml_result)

if __name__ == '__main__':
    app.run(debug=True)
