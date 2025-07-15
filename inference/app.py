from flask import Flask, request, jsonify
from toxic_text_scanner import ToxicTextScanner
from config import CONFIG

app = Flask(__name__)

# Initialize the scanner
scanner = ToxicTextScanner(model_path=CONFIG['model_path'], vectorizer_path=CONFIG['vectorizer_path'])
if not scanner.load_model():
    raise FileNotFoundError("Model or vectorizer not found. Please run the training script first.")

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        result = scanner.classify(text)
        return jsonify({
            'label': result['label'],
            'confidence': result['confidence'],
            'explanation': result['explanation']
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=CONFIG['port'], debug=CONFIG['debug'])