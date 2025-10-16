#!/usr/bin/env python3
"""
Simple Flask App using the single-file code fixer
"""

from flask import Flask, render_template, request, jsonify
from simple_code_fixer import SimpleCodeFixer

app = Flask(__name__)

# Initialize the code fixer (loads dataset and trains models)
print("Initializing Simple Code Fixer...")
fixer = SimpleCodeFixer()
print("Code fixer ready!")

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze code and return results"""
    try:
        data = request.get_json()
        code = data.get('code', '')
        
        if not code.strip():
            return jsonify({'error': 'No code provided'}), 400
        
        # Analyze the code
        result = fixer.analyze_code(code)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """Retrain the models (optional endpoint)"""
    try:
        # Reinitialize the fixer to retrain models
        global fixer
        fixer = SimpleCodeFixer()
        
        return jsonify({
            'success': True,
            'message': 'Models retrained successfully!'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SIMPLE CODE FIXER - WEB INTERFACE")
    print("="*60)
    print("Starting web server...")
    print("🌐 Open your browser and go to: http://localhost:8080")
    print("📱 Or try: http://127.0.0.1:8080")
    print("🛑 Press Ctrl+C to stop the server")
    print("="*60)
    
    app.run(host='0.0.0.0', port=8080, debug=False)
