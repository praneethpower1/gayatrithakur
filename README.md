# Simple Code Fixer

A single-file AI-powered tool for detecting and fixing bugs in Python code, including quantum computing frameworks. Loads the Kaggle dataset directly and trains ML models in real-time.

## Features

- **Single File Solution**: Everything in one Python file
- **Direct Kaggle Dataset**: Loads from CSV, no external .pkl files
- **Real-time Training**: ML models train on startup
- **Universal Code Analysis**: Works with any Python code
- **Quantum Computing Support**: Qiskit, Cirq, and PennyLane detection
- **AI-Powered**: Machine learning trained on 1000+ examples
- **Intelligent Fixes**: Smart code corrections with confidence scores
- **Web Interface**: Clean, fast browser-based tool

## Quick Start

1. Install dependencies:
   ```bash
   pip install flask scikit-learn pandas numpy
   ```

2. Run the tool:
   ```bash
   # Command line version
   python simple_code_fixer.py
   
   # Web interface
   python simple_app.py
   ```

3. Open `http://localhost:8080` in your browser (for web version)

## Usage

### Command Line
```bash
python simple_code_fixer.py
```

### Web Interface
- Paste your code and click "Analyze Code"
- View highlighted errors and suggested fixes
- Copy the corrected code

### Python API
```python
from simple_code_fixer import SimpleCodeFixer

fixer = SimpleCodeFixer()
result = fixer.analyze_code("your buggy code here")
print(result['fixed_code'])
```

## Supported Code Types

- **Python**: General syntax, logic, and import errors
- **Qiskit**: Quantum circuits, gates, measurements
- **Cirq**: Quantum circuits and operations
- **PennyLane**: Quantum functions and devices

## How It Works

1. **Dataset Loading**: Loads 1000 examples from `code_bug_fix_pairs.csv`
2. **Data Cleaning**: Filters to 889 valid bug-fix pairs
3. **Feature Extraction**: Extracts 50+ features per code snippet
4. **ML Training**: Trains Random Forest models in real-time
5. **Bug Detection**: Analyzes code for 6+ bug types
6. **Code Fixing**: Generates corrected code with fixes

## AI Model

- **Training Data**: Full Kaggle dataset (1000 examples)
- **Algorithm**: Random Forest (100 trees for general, 50 for quantum)
- **Features**: 50+ code analysis features
- **Bug Types**: 6+ categories with confidence scoring
- **Real-time**: No pre-trained files, trains on startup

## File Structure

```
gt2/
├── simple_code_fixer.py    # Complete solution (34KB)
├── simple_app.py          # Web interface (2KB)
├── code_bug_fix_pairs.csv # Kaggle dataset (270KB)
├── templates/index.html   # Web UI (15KB)
└── README.md             # Documentation
```

## API

### SimpleCodeFixer.analyze_code(code)
Returns:
```python
{
    'bugs': [...],           # Detected issues
    'fixes': [...],          # Suggested corrections
    'fixed_code': '...',     # Corrected code
    'is_quantum': bool,      # Quantum code detection
    'framework': '...'       # Framework (qiskit/cirq/pennylane)
}
```

## Performance

- **Training**: 1000 examples, real-time training
- **Speed**: Fast analysis and fixes
- **Reliability**: Self-contained, no external dependencies
- **Coverage**: 6+ bug types, quantum frameworks

## License

MIT License - Open source and free to use.