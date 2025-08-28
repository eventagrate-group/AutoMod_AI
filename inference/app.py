from flask import Flask, request, jsonify
from toxic_text_scanner import ToxicTextScanner
import os
from config import CONFIG

app = Flask(__name__)
# Set NLTK data path
os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')
scanner = ToxicTextScanner()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Provide a non-empty 'text' string"}), 400

    result = scanner.classify(text)

    # If the scanner returned an error, pass it through
    if isinstance(result, dict) and "error" in result:
        return jsonify(result), 500

    # Normal successful response
    return jsonify(result), 200

if __name__ == "__main__":
    # Useful for local debugging; production uses gunicorn/PM2
    port = CONFIG.get('port', 5001)  # Default to 5001 if not in CONFIG
    debug = CONFIG.get('debug', False)  # Default to False
    app.run(host="0.0.0.0", port=port, debug=debug)