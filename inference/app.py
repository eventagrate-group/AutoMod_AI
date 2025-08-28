from flask import Flask, request, jsonify
from toxic_text_scanner import ToxicTextScanner
import os
from config import CONFIG
import json

app = Flask(__name__)
# Set NLTK data path
os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')
scanner = ToxicTextScanner()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/classify", methods=["POST"])
def classify():
    try:
        data = request.get_json(silent=True, force=True) or {}
        text = data.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Provide a non-empty 'text' string"}), 400
        # Normalize Unicode to handle encoding issues
        text = text.encode('utf-8').decode('utf-8')
        result = scanner.classify(text)
        # If the scanner returned an error, pass it through
        if isinstance(result, dict) and "error" in result:
            return jsonify(result), 500
        return jsonify(result), 200
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format"}), 400
    except Exception as e:
        return jsonify({"error": f"Request processing failed: {str(e)}"}), 500

if __name__ == "__main__":
    # Useful for local debugging; production uses gunicorn/PM2
    port = CONFIG.get('port', 5001)  # Default to 5001 if not in CONFIG
    debug = CONFIG.get('debug', False)  # Default to False
    app.run(host="0.0.0.0", port=port, debug=debug)