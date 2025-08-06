import re
import ast
import numpy as np
import time
from collections import Counter
import math
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound
from typing import Dict, List, Tuple, Optional
import os
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedCodeAnalyzer:
    def __init__(self):
        self.feature_cache = {}
        self.cache_lock = threading.Lock()
        
        # Initialize models
        self.neural_model = None
        self.scaler = None
        self.label_encoder = None
        
        # Performance optimizations
        self.compiled_patterns = {
            'function_calls': re.compile(r'\b\w+\s*\('),
            'method_calls': re.compile(r'\.\w+'),
            'camel_case': re.compile(r'[A-Z][a-z]+'),
            'snake_case': re.compile(r'[a-z]+_[a-z]+'),
            'variables': re.compile(r'\b[a-zA-Z_]\w*\s*='),
            'add_assign': re.compile(r'\b[a-zA-Z_]\w*\s*\+='),
            'sub_assign': re.compile(r'\b[a-zA-Z_]\w*\s*-='),
            'floats': re.compile(r'\b\d+\.\d+'),
            'integers': re.compile(r'\b\d+'),
            'mixed_case': re.compile(r'[a-z][A-Z]'),
            'multiple_spaces': re.compile(r'\s{2,}'),
            'tabs': re.compile(r'\t'),
            'words': re.compile(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'),
            'comments': re.compile(r'#.*$|/\*.*?\*/|//.*$', re.MULTILINE | re.DOTALL),
            'strings': re.compile(r'"[^"]*"|\'[^\']*\''),
            'imports': re.compile(r'^(import|from)\s+', re.MULTILINE),
            'classes': re.compile(r'^class\s+', re.MULTILINE),
            'functions': re.compile(r'^def\s+', re.MULTILINE),
            'decorators': re.compile(r'^@\w+', re.MULTILINE),
            'async_def': re.compile(r'^async\s+def', re.MULTILINE),
            'type_hints': re.compile(r':\s*\w+(\[\w+\])?'),
            'f_strings': re.compile(r'f["\']'),
            'list_comprehensions': re.compile(r'\[.*for.*in.*\]'),
            'lambda_functions': re.compile(r'lambda\s+'),
            'generator_expressions': re.compile(r'\(.*for.*in.*\)'),
            'context_managers': re.compile(r'with\s+'),
            'exception_handling': re.compile(r'try:|except|finally'),
            'assertions': re.compile(r'assert\s+'),
            'docstrings': re.compile(r'""".*?"""|\'\'\'.*?\'\'\'', re.DOTALL),
            'magic_methods': re.compile(r'__\w+__'),
            'private_methods': re.compile(r'_\w+'),
            'constants': re.compile(r'^[A-Z_][A-Z0-9_]*\s*=', re.MULTILINE)
        }
        
        # Load models in background
        self._load_models()
        print("Enhanced Code Analyzer initialized (Comprehensive Analysis)")

    def _load_models(self):
        """Load neural models if available"""
        try:
            # Try to load improved models first
            if os.path.exists('models/improved_neural_model.pkl'):
                self.neural_model = joblib.load('models/improved_neural_model.pkl')
                print("Loaded improved neural model")
            elif os.path.exists('models/simple_neural_model.pkl'):
                self.neural_model = joblib.load('models/simple_neural_model.pkl')
                print("Loaded simple neural model")
                
            if os.path.exists('models/improved_scaler.pkl'):
                self.scaler = joblib.load('models/improved_scaler.pkl')
                print("Loaded improved scaler")
            elif os.path.exists('models/simple_neural_scaler.pkl'):
                self.scaler = joblib.load('models/simple_neural_scaler.pkl')
                print("Loaded simple neural scaler")
                
            if os.path.exists('models/improved_label_encoder.pkl'):
                self.label_encoder = joblib.load('models/improved_label_encoder.pkl')
                print("Loaded improved label encoder")
            elif os.path.exists('models/simple_neural_label_encoder.pkl'):
                self.label_encoder = joblib.load('models/simple_neural_label_encoder.pkl')
                print("Loaded simple neural label encoder")
            
            print("Enhanced models loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load neural models: {e}")
            print("Falling back to traditional features only")

    @lru_cache(maxsize=2000)
    def extract_features_fast(self, code: str) -> List[float]:
        start_time = time.time()
        
        # Basic features (optimized)
        basic_features = self._extract_basic_features(code)
        
        # Advanced features (optimized)
        advanced_features = self._extract_advanced_features_cached(code)
        
        # Style features (optimized)
        style_features = self._extract_style_features(code)
        
        # Enhanced features (optimized)
        enhanced_features = self._extract_enhanced_features(code)
        
        # Comprehensive analysis features
        comprehensive_features = self._extract_comprehensive_features(code)
        
        # Combine all features
        features = basic_features + advanced_features + style_features + enhanced_features + comprehensive_features
        
        # Always return exactly 80 features
        if len(features) < 80:
            features += [0.0] * (80 - len(features))
        elif len(features) > 80:
            features = features[:80]
        
        elapsed = time.time() - start_time
        print(f"Enhanced feature extraction completed in {elapsed:.3f}s (features: {len(features)})")
        return features

    def _extract_basic_features(self, code: str) -> List[float]:
        """Extract basic code statistics (optimized)"""
        features = []
        
        # Length features
        features.append(len(code))
        features.append(code.count('\n'))
        features.append(len(code.split()))
        
        # Character distribution (optimized)
        alpha_count = sum(1 for c in code if c.isalpha())
        digit_count = sum(1 for c in code if c.isdigit())
        space_count = sum(1 for c in code if c.isspace())
        bracket_count = sum(1 for c in code if c in '{}[]()')
        punct_count = sum(1 for c in code if c in ';:,.')
        
        features.extend([alpha_count, digit_count, space_count, bracket_count, punct_count])
        
        # Line statistics (optimized)
        lines = code.split('\n')
        line_count = len(lines)
        if line_count > 0:
            line_lengths = [len(line) for line in lines]
            features.extend([line_count, np.mean(line_lengths), np.std(line_lengths)])
        else:
            features.extend([0, 0, 0])
        
        return features

    def _extract_advanced_features_cached(self, code: str) -> List[float]:
        """Extract advanced code features with caching (optimized)"""
        features = []
        
        # Comment analysis (optimized)
        lines = code.split('\n')
        comment_lines = sum(1 for line in lines if line.strip().startswith(('#', '//', '/*', '*')))
        features.append(comment_lines)
        features.append(comment_lines / max(len(lines), 1))
        
        # Function and class analysis (optimized)
        try:
            tree = ast.parse(code)
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            classes = len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            imports = len([node for node in ast.walk(tree) if isinstance(node, ast.Import)])
            imports_from = len([node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)])
            calls = len([node for node in ast.walk(tree) if isinstance(node, ast.Call)])
            assignments = len([node for node in ast.walk(tree) if isinstance(node, ast.Assign)])
            loops = len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))])
            conditionals = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
        except:
            functions = classes = imports = imports_from = calls = assignments = loops = conditionals = 0
        
        features.extend([functions, classes, imports, imports_from, calls, assignments, loops, conditionals])
        
        # Complexity metrics (optimized)
        features.append(code.count('if') + code.count('elif') + code.count('else'))
        features.append(code.count('for') + code.count('while'))
        features.append(code.count('try') + code.count('except') + code.count('finally'))
        
        return features

    def _extract_style_features(self, code: str) -> List[float]:
        """Extract code style and formatting features (optimized)"""
        features = []
        
        # Indentation analysis (optimized)
        lines = code.split('\n')
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)
        
        if indent_levels:
            features.extend([np.mean(indent_levels), np.std(indent_levels), max(indent_levels)])
        else:
            features.extend([0, 0, 0])
        
        # Naming conventions (optimized)
        words = self.compiled_patterns['words'].findall(code)
        camel_case = sum(1 for word in words if re.match(r'^[a-z]+[A-Z]', word))
        snake_case = sum(1 for word in words if '_' in word)
        pascal_case = sum(1 for word in words if re.match(r'^[A-Z][a-zA-Z0-9]*$', word))
        
        features.extend([camel_case, snake_case, pascal_case])
        
        # Code structure (optimized)
        features.append(code.count('def '))
        features.append(code.count('class '))
        features.append(code.count('import '))
        features.append(code.count('from '))
        
        return features

    def _extract_enhanced_features(self, code: str) -> List[float]:
        """Extract enhanced features that simulate neural-like patterns (optimized)"""
        features = []
        
        # Code complexity metrics (optimized with compiled patterns)
        features.append(len(self.compiled_patterns['function_calls'].findall(code)))
        features.append(len(self.compiled_patterns['method_calls'].findall(code)))
        features.append(len(self.compiled_patterns['camel_case'].findall(code)))
        features.append(len(self.compiled_patterns['snake_case'].findall(code)))
        
        # Code patterns (optimized)
        patterns = ['return', 'print', 'assert', 'raise', 'with', 'async', 'await']
        for pattern in patterns:
            features.append(code.count(pattern))
        
        # Variable patterns (optimized)
        features.append(len(self.compiled_patterns['variables'].findall(code)))
        features.append(len(self.compiled_patterns['add_assign'].findall(code)))
        features.append(len(self.compiled_patterns['sub_assign'].findall(code)))
        
        # String patterns (optimized)
        features.append(code.count('"'))
        features.append(code.count("'"))
        features.append(code.count('f"'))
        features.append(code.count("f'"))
        
        # Number patterns (optimized)
        features.append(len(self.compiled_patterns['floats'].findall(code)))
        features.append(len(self.compiled_patterns['integers'].findall(code)))
        
        # Control flow patterns (optimized)
        control_patterns = ['break', 'continue', 'pass']
        for pattern in control_patterns:
            features.append(code.count(pattern))
        
        # Error handling (optimized)
        error_patterns = ['except', 'finally', 'else:']
        for pattern in error_patterns:
            features.append(code.count(pattern))
        
        # Documentation (optimized)
        doc_patterns = ['"""', "'''", 'TODO', 'FIXME', 'HACK']
        for pattern in doc_patterns:
            features.append(code.count(pattern))
        
        # Modern Python features (optimized)
        modern_patterns = ['lambda', 'yield', 'generator', 'decorator']
        for pattern in modern_patterns:
            features.append(code.count(pattern))
        
        # Code quality indicators (optimized)
        features.append(len(self.compiled_patterns['mixed_case'].findall(code)))
        features.append(len(self.compiled_patterns['multiple_spaces'].findall(code)))
        features.append(len(self.compiled_patterns['tabs'].findall(code)))
        
        return features

    def _extract_comprehensive_features(self, code: str) -> List[float]:
        """Extract comprehensive analysis features"""
        features = []
        
        # Advanced features
        features.append(len(self.compiled_patterns['type_hints'].findall(code)))
        features.append(len(self.compiled_patterns['f_strings'].findall(code)))
        features.append(len(self.compiled_patterns['list_comprehensions'].findall(code)))
        features.append(len(self.compiled_patterns['lambda_functions'].findall(code)))
        features.append(len(self.compiled_patterns['generator_expressions'].findall(code)))
        features.append(len(self.compiled_patterns['context_managers'].findall(code)))
        features.append(len(self.compiled_patterns['exception_handling'].findall(code)))
        features.append(len(self.compiled_patterns['assertions'].findall(code)))
        features.append(len(self.compiled_patterns['docstrings'].findall(code)))
        features.append(len(self.compiled_patterns['magic_methods'].findall(code)))
        features.append(len(self.compiled_patterns['private_methods'].findall(code)))
        features.append(len(self.compiled_patterns['constants'].findall(code)))
        
        # Code entropy (complexity measure)
        char_freq = Counter(code)
        total_chars = len(code)
        if total_chars > 0:
            entropy = -sum((freq/total_chars) * math.log2(freq/total_chars) for freq in char_freq.values())
            features.append(entropy)
        else:
            features.append(0)
        
        # Unique identifier ratio
        words = self.compiled_patterns['words'].findall(code)
        unique_words = len(set(words))
        features.append(unique_words / max(len(words), 1))
        
        # Function complexity (average function length)
        try:
            tree = ast.parse(code)
            functions = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
            if functions > 0:
                features.append(len(code) / functions)
            else:
                features.append(0)
        except:
            features.append(0)
        
        return features

    def analyze_code_comprehensive(self, code: str) -> Dict:
        """Comprehensive code analysis with detailed insights"""
        analysis = {}
        
        # Basic metrics
        lines = code.split('\n')
        analysis['basic_metrics'] = {
            'total_lines': len(lines),
            'code_lines': len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            'comment_lines': len([line for line in lines if line.strip().startswith('#')]),
            'blank_lines': len([line for line in lines if not line.strip()]),
            'total_characters': len(code),
            'average_line_length': np.mean([len(line) for line in lines if line.strip()]) if lines else 0
        }
        
        # Complexity analysis
        try:
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            analysis['complexity'] = {
                'functions': len(functions),
                'classes': len(classes),
                'imports': len([node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]),
                'function_calls': len([node for node in ast.walk(tree) if isinstance(node, ast.Call)]),
                'assignments': len([node for node in ast.walk(tree) if isinstance(node, ast.Assign)]),
                'loops': len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]),
                'conditionals': len([node for node in ast.walk(tree) if isinstance(node, ast.If)]),
                'exceptions': len([node for node in ast.walk(tree) if isinstance(node, (ast.Try, ast.ExceptHandler))])
            }
        except:
            analysis['complexity'] = {
                'functions': 0, 'classes': 0, 'imports': 0, 'function_calls': 0,
                'assignments': 0, 'loops': 0, 'conditionals': 0, 'exceptions': 0
            }
        
        # Style analysis
        analysis['style'] = {
            'indentation_consistency': self._analyze_indentation(code),
            'naming_conventions': self._analyze_naming_conventions(code),
            'comment_ratio': analysis['basic_metrics']['comment_lines'] / max(analysis['basic_metrics']['total_lines'], 1),
            'line_length_variation': np.std([len(line) for line in lines if line.strip()]) if lines else 0
        }
        
        # Language detection removed - handled by enhanced_app.py with Pygments
        
        # Code quality indicators
        analysis['quality'] = {
            'has_docstrings': bool(self.compiled_patterns['docstrings'].findall(code)),
            'has_type_hints': bool(self.compiled_patterns['type_hints'].findall(code)),
            'has_error_handling': bool(self.compiled_patterns['exception_handling'].findall(code)),
            'has_assertions': bool(self.compiled_patterns['assertions'].findall(code)),
            'uses_modern_features': bool(self.compiled_patterns['f_strings'].findall(code) or 
                                       self.compiled_patterns['lambda_functions'].findall(code))
        }
        
        return analysis

    def _analyze_indentation(self, code: str) -> str:
        """Analyze indentation consistency"""
        lines = code.split('\n')
        indent_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indent_levels.append(indent)
        
        if not indent_levels:
            return "No indentation"
        
        if len(set(indent_levels)) == 1:
            return "Consistent"
        elif len(set(indent_levels)) <= 3:
            return "Mostly consistent"
        else:
            return "Inconsistent"

    def _analyze_naming_conventions(self, code: str) -> Dict:
        """Analyze naming conventions"""
        words = self.compiled_patterns['words'].findall(code)
        
        camel_case = sum(1 for word in words if re.match(r'^[a-z]+[A-Z]', word))
        snake_case = sum(1 for word in words if '_' in word)
        pascal_case = sum(1 for word in words if re.match(r'^[A-Z][a-zA-Z0-9]*$', word))
        
        total = len(words)
        if total == 0:
            return {'camel_case': 0, 'snake_case': 0, 'pascal_case': 0, 'dominant': 'none'}
        
        conventions = {
            'camel_case': camel_case / total,
            'snake_case': snake_case / total,
            'pascal_case': pascal_case / total
        }
        
        dominant = max(conventions, key=conventions.get)
        conventions['dominant'] = dominant
        
        return conventions

    def predict(self, code: str) -> Dict:
        """Make prediction using neural model with comprehensive analysis"""
        start_time = time.time()
        features = self.extract_features_fast(code)
        
        # Get comprehensive analysis
        comprehensive_analysis = self.analyze_code_comprehensive(code)
        
        if self.neural_model is not None and self.scaler is not None:
            try:
                # Scale features
                features_scaled = self.scaler.transform([features])
                
                # Make prediction
                prediction = self.neural_model.predict(features_scaled)[0]
                probability = self.neural_model.predict_proba(features_scaled)[0]
                
                elapsed = time.time() - start_time
                return {
                    'prediction': 'AI' if prediction == 1 else 'Human',
                    'confidence': float(max(probability)),
                    'ai_probability': float(probability[1] if len(probability) > 1 else probability[0]),
                    'human_probability': float(probability[0] if len(probability) > 1 else 1 - probability[0]),
                    'features_used': len(features),
                    'neural_features': 80,
                    'processing_time': elapsed,
                    'comprehensive_analysis': comprehensive_analysis,
                    'model_type': 'enhanced_neural'
                }
            except Exception as e:
                print(f"Error in neural prediction: {e}")
        
        # Fallback to rule-based prediction
        return self._rule_based_prediction(code, features, comprehensive_analysis, time.time() - start_time)

    def _rule_based_prediction(self, code: str, features: List[float], analysis: Dict, elapsed: float) -> Dict:
        """Rule-based prediction as fallback with comprehensive analysis"""
        # Enhanced heuristics
        ai_indicators = 0
        human_indicators = 0
        
        # Check for AI-like patterns
        if features[0] > 1000:  # Long code
            ai_indicators += 1
        if features[1] > 50:  # Many lines
            ai_indicators += 1
        if features[2] > 200:  # Many words
            ai_indicators += 1
        if analysis['style']['comment_ratio'] < 0.1:  # Few comments
            ai_indicators += 1
        if analysis['complexity']['functions'] > 5:  # Many functions
            ai_indicators += 1
        
        # Check for human-like patterns
        if 50 <= features[0] <= 500:  # Moderate length
            human_indicators += 1
        if features[1] < 30:  # Fewer lines
            human_indicators += 1
        if analysis['style']['comment_ratio'] > 0.2:  # More comments
            human_indicators += 1
        if analysis['style']['indentation_consistency'] == 'Consistent':
            human_indicators += 1
        if analysis['quality']['has_docstrings']:
            human_indicators += 1
        
        total_indicators = ai_indicators + human_indicators
        if total_indicators == 0:
            confidence = 0.5
            prediction = 'Human'
        else:
            confidence = max(ai_indicators, human_indicators) / total_indicators
            prediction = 'AI' if ai_indicators > human_indicators else 'Human'
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'ai_probability': confidence if prediction == 'AI' else 1 - confidence,
            'human_probability': confidence if prediction == 'Human' else 1 - confidence,
            'features_used': len(features),
            'neural_features': 80,
            'processing_time': elapsed,
            'comprehensive_analysis': analysis,
            'model_type': 'enhanced_rule_based'
        } 