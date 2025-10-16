#!/usr/bin/env python3
"""
Simple Code Fixer - Single file with built-in Kaggle dataset training
No external .pkl files needed - everything happens in one file!
"""

import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available.")

class SimpleCodeFixer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.quantum_model = None
        self.quantum_vectorizer = None
        self.training_data = None
        self.quantum_training_data = None
        
        # Load and train models automatically
        self.load_kaggle_dataset()
        self.train_models()
    
    def load_kaggle_dataset(self):
        """Load Kaggle dataset directly from CSV"""
        print("Loading Kaggle dataset from code_bug_fix_pairs.csv...")
        
        try:
            df = pd.read_csv('code_bug_fix_pairs.csv')
            print(f"Dataset loaded: {df.shape[0]} examples")
            
            # Clean the data
            df = df.dropna(subset=['buggy_code', 'fixed_code'])
            df['buggy_code'] = df['buggy_code'].astype(str)
            df['fixed_code'] = df['fixed_code'].astype(str)
            
            # Filter out very short or identical codes
            df = df[df['buggy_code'].str.len() > 10]
            df = df[df['fixed_code'].str.len() > 10]
            df = df[df['buggy_code'] != df['fixed_code']]
            
            print(f"After cleaning: {len(df)} valid examples")
            
            # Create training examples
            self.training_data = []
            for idx, row in df.iterrows():
                buggy_code = row['buggy_code'].strip()
                fixed_code = row['fixed_code'].strip()
                bug_type = self._classify_bug_type(buggy_code, fixed_code)
                
                self.training_data.append({
                    'buggy_code': buggy_code,
                    'fixed_code': fixed_code,
                    'bug_type': bug_type
                })
            
            print(f"Created {len(self.training_data)} training examples")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            self.training_data = []
    
    def _classify_bug_type(self, buggy_code, fixed_code):
        """Classify bug type from buggy and fixed code"""
        buggy_lower = buggy_code.lower()
        fixed_lower = fixed_code.lower()
        
        # Missing colon
        if ':' not in buggy_code and ':' in fixed_code:
            return 'missing_colon'
        
        # Missing import
        if 'import' not in buggy_lower and 'import' in fixed_lower:
            return 'missing_import'
        
        # Syntax error (brackets, parentheses)
        if (buggy_code.count('(') != fixed_code.count('(') or 
            buggy_code.count('[') != fixed_code.count('[') or
            buggy_code.count('{') != fixed_code.count('{')):
            return 'syntax_error'
        
        # Indentation error
        if len(buggy_code.split('\n')) == len(fixed_code.split('\n')):
            buggy_lines = buggy_code.split('\n')
            fixed_lines = fixed_code.split('\n')
            for i, (b_line, f_line) in enumerate(zip(buggy_lines, fixed_lines)):
                if b_line.strip() == f_line.strip() and b_line != f_line:
                    return 'indentation_error'
        
        # Variable name error
        if len(buggy_code.split()) != len(fixed_code.split()):
            return 'variable_error'
        
        # Logic error
        if 'if' in buggy_lower and 'if' in fixed_lower:
            return 'logic_error'
        
        # Default
        return 'general_bug'
    
    def _extract_features(self, code):
        """Extract features from code for ML model"""
        features = []
        
        # Basic features
        features.append(f"code_length_{len(code)}")
        features.append(f"line_count_{len(code.split())}")
        features.append(f"char_count_{len(code)}")
        
        # Language patterns
        if 'def ' in code:
            features.append('has_function_definition')
        if 'class ' in code:
            features.append('has_class_definition')
        if 'import ' in code:
            features.append('has_imports')
        if 'if ' in code:
            features.append('has_conditionals')
        if 'for ' in code or 'while ' in code:
            features.append('has_loops')
        if 'try:' in code:
            features.append('has_error_handling')
        if 'return ' in code:
            features.append('has_return_statements')
        if 'print(' in code:
            features.append('has_print_statements')
        if 'lambda ' in code:
            features.append('has_lambda')
        if 'yield ' in code:
            features.append('has_yield')
        
        # Syntax patterns
        if ':' not in code:
            features.append('missing_colon')
        if '(' in code and ')' not in code:
            features.append('unclosed_parentheses')
        if '[' in code and ']' not in code:
            features.append('unclosed_brackets')
        if '{' in code and '}' not in code:
            features.append('unclosed_braces')
        if '"' in code and code.count('"') % 2 != 0:
            features.append('unclosed_quotes')
        if "'" in code and code.count("'") % 2 != 0:
            features.append('unclosed_single_quotes')
        
        # Common bug indicators
        if '=' in code and '==' not in code:
            features.append('assignment_in_condition')
        if 'print(' in code and 'import' not in code:
            features.append('missing_print_import')
        if 'len(' in code and 'import' not in code:
            features.append('missing_len_import')
        if 'range(' in code and 'import' not in code:
            features.append('missing_range_import')
        
        # Indentation patterns
        lines = code.split('\n')
        indent_levels = []
        for line in lines:
            if line.strip():
                indent_levels.append(len(line) - len(line.lstrip()))
        
        if indent_levels:
            features.append(f"max_indent_{max(indent_levels)}")
            features.append(f"avg_indent_{sum(indent_levels)/len(indent_levels):.1f}")
        
        # String patterns
        if 'f"' in code or "f'" in code:
            features.append('has_f_strings')
        if 'r"' in code or "r'" in code:
            features.append('has_raw_strings')
        
        # Quantum patterns
        quantum_keywords = ['qiskit', 'cirq', 'pennylane', 'qml', 'quantum', 'qubit']
        if any(keyword in code.lower() for keyword in quantum_keywords):
            features.append('is_quantum_code')
        
        return ' '.join(features)
    
    def train_models(self):
        """Train both general and quantum models"""
        if not ML_AVAILABLE:
            print("ML libraries not available. Cannot train models.")
            return
        
        # Train general model
        if self.training_data:
            print("Training general ML model on Kaggle dataset...")
            self._train_general_model()
        
        # Train quantum model
        print("Training quantum ML model...")
        self._train_quantum_model()
    
    def _train_general_model(self):
        """Train general model on Kaggle dataset"""
        # Prepare training data
        X = []
        y = []
        
        for example in self.training_data:
            buggy_code = example['buggy_code']
            bug_type = example['bug_type']
            
            features = self._extract_features(buggy_code)
            X.append(features)
            y.append(bug_type)
        
        print(f"Training on {len(X)} examples with {len(set(y))} bug types")
        print(f"Bug types: {set(y)}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Vectorize text features
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_vectorized, y_encoded)
        
        print("✅ General model trained successfully!")
    
    def _train_quantum_model(self):
        """Train quantum-specific model"""
        # Create quantum training data
        self.quantum_training_data = [
            {
                'buggy_code': 'qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)',
                'fixed_code': 'from qiskit import QuantumCircuit, execute, Aer\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\nqc.measure_all()\nbackend = Aer.get_backend(\'qasm_simulator\')\njob = execute(qc, backend, shots=1024)',
                'bug_type': 'missing_quantum_import'
            },
            {
                'buggy_code': 'from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.H(0)\nqc.CX(0, 1)',
                'fixed_code': 'from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)',
                'bug_type': 'incorrect_gate_name'
            },
            {
                'buggy_code': 'import cirq\nqubits = cirq.LineQubit.range(2)\ncircuit = cirq.Circuit()\ncircuit.append(cirq.H(qubits[0]))',
                'fixed_code': 'import cirq\nqubits = cirq.LineQubit.range(2)\ncircuit = cirq.Circuit()\ncircuit.append(cirq.H(qubits[0]))\ncircuit.append(cirq.measure(qubits, key=\'result\'))',
                'bug_type': 'missing_measurement'
            },
            {
                'buggy_code': 'import pennylane as qml\ndev = qml.device(\'default.qubit\', wires=2)\n@qml.qnode(dev)\ndef circuit():\n    qml.hadamard(0)\n    return qml.probs([0, 1])',
                'fixed_code': 'import pennylane as qml\ndev = qml.device(\'default.qubit\', wires=2)\n@qml.qnode(dev)\ndef circuit():\n    qml.Hadamard(wires=0)\n    return qml.probs(wires=[0, 1])',
                'bug_type': 'incorrect_gate_name'
            },
            {
                'buggy_code': 'qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\nqc.measure_all()',
                'fixed_code': 'from qiskit import QuantumCircuit, execute, Aer\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)\nqc.measure_all()\nbackend = Aer.get_backend(\'qasm_simulator\')\njob = execute(qc, backend, shots=1024)',
                'bug_type': 'missing_quantum_import'
            }
        ]
        
        # Prepare training data
        X = []
        y = []
        
        for example in self.quantum_training_data:
            buggy_code = example['buggy_code']
            bug_type = example['bug_type']
            
            features = self._extract_features(buggy_code)
            X.append(features)
            y.append(bug_type)
        
        print(f"Training quantum model on {len(X)} examples with {len(set(y))} bug types")
        print(f"Quantum bug types: {set(y)}")
        
        # Vectorize text features
        self.quantum_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        X_vectorized = self.quantum_vectorizer.fit_transform(X)
        
        # Train model
        self.quantum_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        self.quantum_model.fit(X_vectorized, y)
        
        print("✅ Quantum model trained successfully!")
    
    def is_quantum_code(self, code):
        """Check if code is quantum computing related"""
        quantum_keywords = [
            'qiskit', 'cirq', 'pennylane', 'qml', 'quantum', 'qubit', 'quantumcircuit',
            'quantumregister', 'classicalregister', 'measure', 'h(', 'x(', 'y(', 'z(',
            'cx(', 'cz(', 'ccx(', 'hadamard', 'pauli', 'gate', 'circuit', 'backend',
            'simulator', 'quantumcomputer', 'quantumdevice', 'quantumstate', 'linequbit',
            'qnode', 'quantumdevice', 'quantumfunction', 'quantumoperation'
        ]
        
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in quantum_keywords)
    
    def detect_framework(self, code):
        """Detect quantum computing framework"""
        code_lower = code.lower()
        
        if 'qiskit' in code_lower or 'quantumcircuit' in code_lower:
            return 'qiskit'
        elif 'cirq' in code_lower or 'linequbit' in code_lower:
            return 'cirq'
        elif 'pennylane' in code_lower or 'qml' in code_lower:
            return 'pennylane'
        else:
            return 'unknown'
    
    def predict_bug_type(self, code):
        """Predict bug type using trained model"""
        if not self.model or not self.vectorizer:
            return "general_bug", 0.5
        
        features = self._extract_features(code)
        X = self.vectorizer.transform([features])
        prediction_encoded = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0].max()
        
        # Decode prediction if label encoder is available
        if self.label_encoder:
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        else:
            prediction = str(prediction_encoded)
        
        return prediction, probability
    
    def predict_quantum_bug_type(self, code):
        """Predict bug type using quantum-specific model"""
        if not self.quantum_model or not self.quantum_vectorizer:
            return "quantum_bug", 0.5
        
        features = self._extract_features(code)
        X = self.quantum_vectorizer.transform([features])
        prediction = self.quantum_model.predict(X)[0]
        probability = self.quantum_model.predict_proba(X)[0].max()
        
        return prediction, probability
    
    def detect_bugs_with_lines(self, code):
        """Detect bugs and return specific line numbers with error details"""
        bugs = []
        lines = code.split('\n')
        
        # Check if it's quantum code
        is_quantum = self.is_quantum_code(code)
        framework = self.detect_framework(code) if is_quantum else 'none'
        
        # Use appropriate ML model
        if is_quantum and self.quantum_model:
            predicted_bug_type, confidence = self.predict_quantum_bug_type(code)
        else:
            predicted_bug_type, confidence = self.predict_bug_type(code)
        
        # Line-by-line analysis
        for i, line in enumerate(lines):
            line_num = i + 1
            line_stripped = line.strip()
            
            if not line_stripped:
                continue
            
            # Missing colon detection
            if any(keyword in line_stripped.lower() for keyword in ['if ', 'def ', 'for ', 'while ', 'else', 'except', 'elif', 'class ']):
                if not line_stripped.endswith(':'):
                    bugs.append({
                        'line': line_num,
                        'type': 'missing_colon',
                        'description': 'Missing colon at end of statement',
                        'severity': 'high',
                        'confidence': confidence,
                        'original_line': line,
                        'error_text': line_stripped
                    })
            
            # Missing closing parenthesis
            if line.count('(') > line.count(')'):
                bugs.append({
                    'line': line_num,
                    'type': 'unclosed_parentheses',
                    'description': 'Missing closing parenthesis',
                    'severity': 'high',
                    'confidence': confidence,
                    'original_line': line,
                    'error_text': line_stripped
                })
            
            # Missing closing bracket
            if line.count('[') > line.count(']'):
                bugs.append({
                    'line': line_num,
                    'type': 'unclosed_brackets',
                    'description': 'Missing closing bracket',
                    'severity': 'high',
                    'confidence': confidence,
                    'original_line': line,
                    'error_text': line_stripped
                })
            
            # Missing closing brace
            if line.count('{') > line.count('}'):
                bugs.append({
                    'line': line_num,
                    'type': 'unclosed_braces',
                    'description': 'Missing closing brace',
                    'severity': 'high',
                    'confidence': confidence,
                    'original_line': line,
                    'error_text': line_stripped
                })
            
            # Unclosed quotes
            if line.count('"') % 2 != 0:
                bugs.append({
                    'line': line_num,
                    'type': 'unclosed_quotes',
                    'description': 'Unclosed double quotes',
                    'severity': 'high',
                    'confidence': confidence,
                    'original_line': line,
                    'error_text': line_stripped
                })
            
            if line.count("'") % 2 != 0:
                bugs.append({
                    'line': line_num,
                    'type': 'unclosed_quotes',
                    'description': 'Unclosed single quotes',
                    'severity': 'high',
                    'confidence': confidence,
                    'original_line': line,
                    'error_text': line_stripped
                })
            
            # Indentation issues
            if line_stripped and not line.startswith(' ') and not line.startswith('\t'):
                prev_lines = lines[:i]
                should_indent = False
                for prev_line in reversed(prev_lines):
                    if prev_line.strip().endswith(':'):
                        should_indent = True
                        break
                    elif prev_line.strip():
                        break
                
                if should_indent and line_stripped and not line_stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'else', 'elif', 'except', 'finally')):
                    bugs.append({
                        'line': line_num,
                        'type': 'indentation_error',
                        'description': 'Incorrect indentation',
                        'severity': 'medium',
                        'confidence': confidence,
                        'original_line': line,
                        'error_text': line_stripped
                    })
            
            # Function definition errors
            if 'def ' in line_stripped and '(' in line_stripped and ')' not in line_stripped:
                bugs.append({
                    'line': line_num,
                    'type': 'function_error',
                    'description': 'Incomplete function definition',
                    'severity': 'high',
                    'confidence': confidence,
                    'original_line': line,
                    'error_text': line_stripped
                })
            
            # Loop errors
            if any(keyword in line_stripped.lower() for keyword in ['for ', 'while ']) and ':' not in line_stripped:
                bugs.append({
                    'line': line_num,
                    'type': 'loop_error',
                    'description': 'Incomplete loop definition',
                    'severity': 'high',
                    'confidence': confidence,
                    'original_line': line,
                    'error_text': line_stripped
                })
        
        # Check for missing imports
        if any(keyword in code.lower() for keyword in ['print(', 'len(', 'range(', 'str(', 'int(', 'float(']):
            if 'import' not in code:
                bugs.append({
                    'line': 1,
                    'type': 'missing_import',
                    'description': 'Missing import statements',
                    'severity': 'medium',
                    'confidence': confidence,
                    'original_line': '',
                    'error_text': 'Missing imports'
                })
        
        # Quantum-specific bug detection
        if is_quantum:
            bugs.extend(self._detect_quantum_bugs(code, lines, framework, confidence))
        
        return bugs, f"Analysis complete (predicted: {predicted_bug_type}, confidence: {confidence:.2f})"
    
    def _detect_quantum_bugs(self, code, lines, framework, confidence):
        """Detect quantum-specific bugs"""
        bugs = []
        
        if framework == 'qiskit':
            # Qiskit specific bugs
            if 'quantumcircuit' in code.lower() and 'from qiskit' not in code:
                bugs.append({
                    'line': 1,
                    'type': 'missing_quantum_import',
                    'description': 'Missing Qiskit import statements',
                    'severity': 'high',
                    'confidence': 0.9,
                    'original_line': '',
                    'error_text': 'Missing Qiskit imports'
                })
            
            if 'h(' in code.lower() and 'h(' not in code:
                for i, line in enumerate(lines):
                    if 'h(' in line.lower():
                        bugs.append({
                            'line': i + 1,
                            'type': 'incorrect_gate_name',
                            'description': 'Incorrect gate name case (should be h not H)',
                            'severity': 'medium',
                            'confidence': 0.8,
                            'original_line': line,
                            'error_text': line.strip()
                        })
            
            if 'quantumcircuit' in code.lower() and 'measure' not in code.lower():
                bugs.append({
                    'line': len(lines),
                    'type': 'missing_measurement',
                    'description': 'Missing measurement operations',
                    'severity': 'high',
                    'confidence': 0.9,
                    'original_line': '',
                    'error_text': 'Missing measurement'
                })
        
        elif framework == 'cirq':
            # Cirq specific bugs
            if 'cirq' in code.lower() and 'import cirq' not in code:
                bugs.append({
                    'line': 1,
                    'type': 'missing_quantum_import',
                    'description': 'Missing Cirq import statement',
                    'severity': 'high',
                    'confidence': 0.9,
                    'original_line': '',
                    'error_text': 'Missing Cirq import'
                })
            
            if 'circuit.append' in code and 'measure' not in code.lower():
                bugs.append({
                    'line': len(lines),
                    'type': 'missing_measurement',
                    'description': 'Missing measurement operations',
                    'severity': 'high',
                    'confidence': 0.9,
                    'original_line': '',
                    'error_text': 'Missing measurement'
                })
        
        elif framework == 'pennylane':
            # PennyLane specific bugs
            if 'qml' in code.lower() and 'import pennylane' not in code:
                bugs.append({
                    'line': 1,
                    'type': 'missing_quantum_import',
                    'description': 'Missing PennyLane import statement',
                    'severity': 'high',
                    'confidence': 0.9,
                    'original_line': '',
                    'error_text': 'Missing PennyLane import'
                })
            
            if 'qml.hadamard' in code.lower():
                for i, line in enumerate(lines):
                    if 'qml.hadamard' in line.lower():
                        bugs.append({
                            'line': i + 1,
                            'type': 'incorrect_gate_name',
                            'description': 'Incorrect gate name (should be qml.Hadamard)',
                            'severity': 'medium',
                            'confidence': 0.8,
                            'original_line': line,
                            'error_text': line.strip()
                        })
        
        return bugs
    
    def suggest_fixes(self, code, bugs):
        """Suggest fixes for detected bugs"""
        fixes = []
        lines = code.split('\n')
        framework = self.detect_framework(code)
        
        for bug in bugs:
            line_num = bug['line'] - 1
            bug_type = bug['type']
            original_line = bug['original_line']
            
            if bug_type == 'missing_colon':
                if line_num < len(lines):
                    fixed_line = lines[line_num] + ':'
                    fixes.append({
                        'line': line_num + 1,
                        'original': original_line,
                        'suggestion': fixed_line,
                        'type': 'colon_fix'
                    })
            
            elif bug_type == 'unclosed_parentheses':
                if line_num < len(lines):
                    fixed_line = lines[line_num] + ')'
                    fixes.append({
                        'line': line_num + 1,
                        'original': original_line,
                        'suggestion': fixed_line,
                        'type': 'parentheses_fix'
                    })
            
            elif bug_type == 'unclosed_brackets':
                if line_num < len(lines):
                    fixed_line = lines[line_num] + ']'
                    fixes.append({
                        'line': line_num + 1,
                        'original': original_line,
                        'suggestion': fixed_line,
                        'type': 'brackets_fix'
                    })
            
            elif bug_type == 'unclosed_braces':
                if line_num < len(lines):
                    fixed_line = lines[line_num] + '}'
                    fixes.append({
                        'line': line_num + 1,
                        'original': original_line,
                        'suggestion': fixed_line,
                        'type': 'braces_fix'
                    })
            
            elif bug_type == 'unclosed_quotes':
                if line_num < len(lines):
                    if '"' in lines[line_num] and lines[line_num].count('"') % 2 != 0:
                        fixed_line = lines[line_num] + '"'
                    elif "'" in lines[line_num] and lines[line_num].count("'") % 2 != 0:
                        fixed_line = lines[line_num] + "'"
                    else:
                        fixed_line = lines[line_num]
                    
                    fixes.append({
                        'line': line_num + 1,
                        'original': original_line,
                        'suggestion': fixed_line,
                        'type': 'quotes_fix'
                    })
            
            elif bug_type == 'indentation_error':
                if line_num < len(lines):
                    # Add proper indentation
                    fixed_line = '    ' + lines[line_num].lstrip()
                    fixes.append({
                        'line': line_num + 1,
                        'original': original_line,
                        'suggestion': fixed_line,
                        'type': 'indentation_fix'
                    })
            
            elif bug_type == 'missing_import':
                fixes.append({
                    'line': 0,
                    'original': '',
                    'suggestion': 'import sys\nimport os',
                    'type': 'import_fix'
                })
            
            # Quantum-specific fixes
            elif bug_type == 'missing_quantum_import':
                if framework == 'qiskit':
                    fixes.append({
                        'line': 0,
                        'original': '',
                        'suggestion': 'from qiskit import QuantumCircuit, execute, Aer',
                        'type': 'quantum_import_fix'
                    })
                elif framework == 'cirq':
                    fixes.append({
                        'line': 0,
                        'original': '',
                        'suggestion': 'import cirq',
                        'type': 'quantum_import_fix'
                    })
                elif framework == 'pennylane':
                    fixes.append({
                        'line': 0,
                        'original': '',
                        'suggestion': 'import pennylane as qml',
                        'type': 'quantum_import_fix'
                    })
            
            elif bug_type == 'incorrect_gate_name':
                if line_num < len(lines):
                    original_line = lines[line_num]
                    if 'h(' in original_line.lower():
                        fixes.append({
                            'line': line_num + 1,
                            'original': original_line,
                            'suggestion': original_line.replace('h(', 'H('),
                            'type': 'gate_name_fix'
                        })
                    elif 'qml.hadamard' in original_line.lower():
                        fixes.append({
                            'line': line_num + 1,
                            'original': original_line,
                            'suggestion': original_line.replace('qml.hadamard', 'qml.Hadamard'),
                            'type': 'gate_name_fix'
                        })
            
            elif bug_type == 'missing_measurement':
                if framework == 'qiskit':
                    fixes.append({
                        'line': len(lines),
                        'original': '',
                        'suggestion': 'qc.measure_all()\nbackend = Aer.get_backend(\'qasm_simulator\')\njob = execute(qc, backend, shots=1024)',
                        'type': 'measurement_fix'
                    })
                elif framework == 'cirq':
                    fixes.append({
                        'line': len(lines),
                        'original': '',
                        'suggestion': 'circuit.append(cirq.measure(qubits, key=\'result\'))',
                        'type': 'measurement_fix'
                    })
        
        return fixes
    
    def generate_fixed_code(self, original_code, bugs, fixes):
        """Generate fixed version of the code"""
        lines = original_code.split('\n')
        
        for fix in fixes:
            line_idx = fix['line'] - 1
            
            if fix['type'] in ['import_fix', 'quantum_import_fix', 'measurement_fix']:
                if line_idx == 0:
                    lines.insert(0, fix['suggestion'])
                else:
                    lines.append(fix['suggestion'])
            elif line_idx < len(lines):
                lines[line_idx] = fix['suggestion']
        
        return '\n'.join(lines)
    
    def analyze_code(self, code):
        """Complete analysis of code"""
        bugs, message = self.detect_bugs_with_lines(code)
        fixes = self.suggest_fixes(code, bugs)
        fixed_code = self.generate_fixed_code(code, bugs, fixes)
        
        is_quantum = self.is_quantum_code(code)
        framework = self.detect_framework(code) if is_quantum else 'none'
        
        return {
            'bugs': bugs,
            'fixes': fixes,
            'fixed_code': fixed_code,
            'message': message,
            'buggy_code': code,
            'is_quantum': is_quantum,
            'framework': framework
        }

def main():
    """Test the simple code fixer"""
    print("="*60)
    print("SIMPLE CODE FIXER - SINGLE FILE WITH KAGGLE DATASET")
    print("="*60)
    
    # Initialize the fixer (this will load dataset and train models)
    print("Initializing code fixer...")
    fixer = SimpleCodeFixer()
    
    # Test with sample code
    test_code = """def factorial(n)
if n == 1
    return 1
else:
    return n * factorial(n-1)"""
    
    print("\nTesting with sample code:")
    print("ORIGINAL CODE:")
    print(test_code)
    
    result = fixer.analyze_code(test_code)
    
    print("\nANALYSIS RESULT:")
    print(f"Message: {result['message']}")
    print(f"Bugs detected: {len(result['bugs'])}")
    
    if result['bugs']:
        print("\nBUGS DETECTED:")
        for bug in result['bugs']:
            print(f"  Line {bug['line']}: {bug['description']} ({bug['type']})")
            print(f"    Error: {bug['error_text']}")
        
        print("\nFIXES APPLIED:")
        for fix in result['fixes']:
            print(f"  Line {fix['line']}: {fix['type']}")
        
        print("\nFIXED CODE:")
        print(result['fixed_code'])
    else:
        print("No bugs detected!")
    
    print(f"\nIs Quantum Code: {result['is_quantum']}")
    print(f"Framework: {result['framework']}")

if __name__ == "__main__":
    main()
