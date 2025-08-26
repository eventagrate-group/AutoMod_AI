from flask import Flask, request, jsonify
from toxic_text_scanner import ToxicTextScanner
import os
import nltk

app = Flask(__name__)
nltk.data.path.append(os.getenv('NLTK_DATA', os.path.join(os.getcwd(), 'nltk_data')))

# Initialize the scanner
try:
    scanner = ToxicTextScanner()
except FileNotFoundError as e:
    app.logger.error(f"Model or vectorizer not found: {e}")
    raise
except Exception as e:
    app.logger.error(f"Failed to initialize scanner: {e}")
    raise

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = scanner.classify_text(text)
        if 'error' in result:
            app.logger.error(f"Classification error: {result['error']}")
            return jsonify({'error': result['error']}), 500
        return jsonify({
            'label': result['label'],
            'confidence': result['confidence'],
            'explanation': result['explanation'],
            'influential_terms': result.get('influential_terms', [])
        }), 200
    except FileNotFoundError:
        app.logger.error("Model or vectorizer not found")
        return jsonify({'error': 'Model or vectorizer not found. Please run the training script first.'}), 500
    except Exception as e:
        app.logger.error(f"Error in /classify: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from config import CONFIG
    app.run(host='0.0.0.0', port=CONFIG['port'], debug=CONFIG['debug'])