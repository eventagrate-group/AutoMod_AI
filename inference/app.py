from flask import Flask, request, jsonify
from toxic_text_scanner import ToxicTextScanner

app = Flask(__name__)

# Initialize the scanner
scanner = ToxicTextScanner()

@app.route('/classify', methods=['POST'])
def classify_text():
    try:
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
    app.run(host='0.0.0.0', port=5001, debug=True)