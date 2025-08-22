from flask import Flask, request, jsonify
from toxic_text_scanner import ToxicTextScanner
import os

app = Flask(__name__)

# Initialize the scanner
try:
    scanner = ToxicTextScanner()
except FileNotFoundError:
    print("Model or vectorizer not found. Ensure model.joblib and vectorizer.joblib exist.")
    raise
except Exception as e:
    print(f"Failed to initialize scanner: {e}")
    raise

# Optional API key (uncomment to enable)
# API_KEY = "y26717caa7fd6549c33e0811f1e6ded6f40c167a7026ac0782e363ab461ee0584"

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        # Optional API key check
        # api_key = request.headers.get('X-API-Key')
        # if not api_key or api_key != API_KEY:
        #     return jsonify({'error': 'Invalid or missing API key'}), 401
        
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = scanner.classify_text(text)
        return jsonify({
            'label': result['label'],
            'confidence': result['confidence'],
            'explanation': result['explanation']
        }), 200
    except FileNotFoundError:
        return jsonify({'error': 'Model or vectorizer not found. Please run the training script first.'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from config import CONFIG
    app.run(host='0.0.0.0', port=CONFIG['port'], debug=CONFIG['debug'])