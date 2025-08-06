import pickle
import numpy as np
import random
import re # Import regex for more robust feature extraction
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import json
import pandas as pd
import io
import aiohttp # Import aiohttp for asynchronous HTTP requests - still needed for CSV fetching
import aiohttp.client_exceptions # For specific error handling
import asyncio # Import asyncio for sleep

# Define the order of features for consistent array conversion
FEATURE_ORDER = [
    'total_chars_no_comments', 'num_lines_no_comments', 'avg_line_length_no_comments',
    'ai_keyword_present', 'keyword_count', 'indent_consistency', 'unique_chars',
    'special_chars', 'alphanum_ratio', 'code_lines_count', 'code_line_ratio',
    'function_defs', 'loop_count', 'conditional_count', 'operator_count',
    'numeric_literal_count', 'string_literal_count', 'avg_word_length_in_code',
    'uppercase_ratio', 'num_imports', 'num_classes', 'unique_word_ratio_no_comments',
    'blank_lines_ratio', 'original_comment_ratio', 'token_count', 'avg_token_length',
    'keyword_density', 'operator_density', 'punctuator_count', 'max_indent_depth',
    'avg_params_per_function', 'halstead_unique_operands', 'halstead_total_operands',
    'cyclomatic_complexity_proxy', 'avg_identifier_length', 'unique_operators',
    'ratio_short_lines', 'combined_literal_count', 'avg_nested_block_depth',
    'func_param_variance', 'ratio_long_lines', 'max_paren_depth',
    'num_distinct_data_types', 'avg_lines_per_function', 'use_of_fstrings_template_literals',
    'num_try_except_blocks', 'ratio_magic_numbers', 'has_tf_keras_boilerplate',
    # New Complexity Features
    'num_logical_operators', 'num_comparison_operators', 'ratio_nested_conditionals'
]

# Constants for probability capping
MAX_OUTPUT_PROB = 0.98 # Maximum allowed probability for any single class (e.g., 98%)
MIN_OUTPUT_PROB = 0.01 # Minimum allowed probability for any single class (e.g., 1%)

# --- 1. Load Pre-trained Models (for reference/initialization, not for weights) ---
# These are loaded to get the structure if needed, but the model will be retrained from scratch.
def load_existing_model_components(model_type):
    """Loads a set of model components (neural_model, scaler, label_encoder) based on type."""
    model_path = f'{model_type}_neural_model.pkl'
    scaler_path = f'{model_type}_scaler.pkl'
    label_encoder_path = f'{model_type}_label_encoder.pkl'

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Loaded {model_type} models successfully.")
        return model, scaler, label_encoder
    except Exception as e:
        print(f"Warning: Could not load {model_type} models: {e}. Returning None.")
        return None, None, None

# --- 2. Feature Extraction Function (for Code Detection) ---
# This function extracts features from a given code snippet.
# These features are designed to help distinguish between AI and human code.
def extract_features(code_snippet):
    """
    Extracts features from a code snippet for AI/human detection.
    Returns a dictionary of named features.
    """
    # Initialize all features to 0.0 or appropriate default
    features_dict = {feature_name: 0.0 for feature_name in FEATURE_ORDER}

    if not isinstance(code_snippet, str) or not code_snippet.strip():
        return features_dict # Return dict of zeros for invalid input
    
    original_lines = code_snippet.split('\n')
    original_num_lines = len(original_lines)

    # Calculate original_comment_ratio from the original snippet
    original_comment_lines = 0
    for line in original_lines:
        stripped_line = line.strip()
        if stripped_line.startswith('#') or stripped_line.startswith('//') or stripped_line.startswith('/*'):
            original_comment_lines += 1
    features_dict['original_comment_ratio'] = original_comment_lines / original_num_lines if original_num_lines > 0 else 0.0

    # Calculate blank_lines_ratio from the original snippet
    blank_lines = sum(1 for line in original_lines if not line.strip())
    features_dict['blank_lines_ratio'] = blank_lines / original_num_lines if original_num_lines > 0 else 0.0


    # Remove comments and string literals for core code analysis
    code_no_comments_and_strings = re.sub(r'#.*$', '', code_snippet, flags=re.MULTILINE)
    code_no_comments_and_strings = re.sub(r'//.*$', '', code_no_comments_and_strings, flags=re.MULTILINE)
    code_no_comments_and_strings = re.sub(r'/\*.*?\*/', '', code_no_comments_and_strings, flags=re.DOTALL)
    code_no_comments_and_strings = re.sub(r'"[^"]*"|\'[^\']*\'', '', code_no_comments_and_strings) # Remove string literals

    lines_of_code = [line.strip() for line in code_no_comments_and_strings.split('\n') if line.strip()]
    
    num_lines_no_comments = len(lines_of_code)
    total_chars_no_comments = len(code_no_comments_and_strings)
    avg_line_length_no_comments = total_chars_no_comments / num_lines_no_comments if num_lines_no_comments > 0 else 0.0

    features_dict['total_chars_no_comments'] = total_chars_no_comments
    features_dict['num_lines_no_comments'] = num_lines_no_comments
    features_dict['avg_line_length_no_comments'] = avg_line_length_no_comments


    # Presence of AI-specific phrases (check original snippet)
    ai_keywords = [
        "as an ai model", "i am a large language model", "i cannot",
        "i do not have", "i can help you with", "i'm an ai",
        "i am a bot", "i do not have access to real-time information",
        "i am programmed to", "i am designed to"
    ]
    features_dict['ai_keyword_present'] = 1.0 if any(keyword in code_snippet.lower() for keyword in ai_keywords) else 0.0

    # Presence of common programming language keywords (in code without comments and strings)
    common_keywords = [
        "def ", "class ", "import ", "if ", "else ", "for ", "while ",
        "function ", "const ", "var ", "let ", "public ", "private ",
        "static ", "void ", "return ", "try ", "catch ", "finally ",
        "int ", "float ", "double ", "char ", "bool ", "string ", "array ", "list ", "dict ", "set " # Added common type keywords
    ]
    features_dict['keyword_count'] = sum(1 for keyword in common_keywords if keyword in code_no_comments_and_strings)

    # Indentation consistency (in code without comments and strings)
    indent_chars = [line[0] for line in lines_of_code if line and line[0] in [' ', '\t']]
    features_dict['indent_consistency'] = 0.0
    if len(indent_chars) > 1:
        first_indent = indent_chars[0]
        consistent_indents = all(c == first_indent for c in indent_chars)
        features_dict['indent_consistency'] = 1.0 if consistent_indents else 0.0

    # Number of unique characters (in code without comments and strings)
    features_dict['unique_chars'] = len(set(list(code_no_comments_and_strings)))

    # Number of special characters (e.g., punctuation, operators) in code without comments and strings
    features_dict['special_chars'] = sum(1 for char in code_no_comments_and_strings if not char.isalnum() and not char.isspace())

    # Ratio of alphanumeric characters to total characters (in code without comments and strings)
    features_dict['alphanum_ratio'] = sum(1 for char in code_no_comments_and_strings if char.isalnum()) / (total_chars_no_comments if total_chars_no_comments > 0 else 1.0)

    # Number of lines with actual code (non-empty, non-comment lines)
    features_dict['code_lines_count'] = num_lines_no_comments
    features_dict['code_line_ratio'] = features_dict['code_lines_count'] / original_num_lines if original_num_lines > 0 else 0.0

    # Number of function/method definitions (in code without comments and strings)
    features_dict['function_defs'] = code_no_comments_and_strings.lower().count("def ") + \
                                     code_no_comments_and_strings.lower().count("function ") + \
                                     code_no_comments_and_strings.lower().count("void ") + \
                                     code_no_comments_and_strings.lower().count("int ") + \
                                     code_no_comments_and_strings.lower().count("string ") # Basic C++/Java function detection

    # Number of loops (in code without comments and strings)
    features_dict['loop_count'] = code_no_comments_and_strings.lower().count("for ") + code_no_comments_and_strings.lower().count("while ")

    # Number of conditional statements (in code without comments and strings)
    features_dict['conditional_count'] = code_no_comments_and_strings.lower().count("if ") + code_no_comments_and_strings.lower().count("else ")

    # Operator Count (in code without comments and strings)
    operators = ['+', '-', '*', '/', '%', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!', '&', '|', '^', '~', '<<', '>>', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '>>=', '<<=']
    features_dict['operator_count'] = sum(code_no_comments_and_strings.count(op) for op in operators)

    # Numeric Literal Count (in code without comments and strings)
    features_dict['numeric_literal_count'] = len(re.findall(r'\b\d+\.?\d*\b', code_no_comments_and_strings))

    # String Literal Count (simple approximation, in code without comments and strings)
    features_dict['string_literal_count'] = len(re.findall(r'"[^"]*"|\'[^\']*\'', code_snippet)) # Count from original snippet

    # Average Word Length in Code (excluding comments and strings)
    words_in_code = re.findall(r'\b\w+\b', code_no_comments_and_strings)
    features_dict['avg_word_length_in_code'] = np.mean([len(word) for word in words_in_code]) if words_in_code else 0.0

    # Ratio of Uppercase Characters (in code without comments and strings)
    uppercase_chars = sum(1 for char in code_no_comments_and_strings if char.isupper())
    total_alphabetic_chars = sum(1 for char in code_no_comments_and_strings if char.isalpha())
    features_dict['uppercase_ratio'] = uppercase_chars / total_alphabetic_chars if total_alphabetic_chars > 0 else 0.0

    # NEW FEATURES
    features_dict['num_imports'] = len(re.findall(r'^\s*(?:import|#include|using)\s+', code_snippet, flags=re.MULTILINE))
    features_dict['num_classes'] = len(re.findall(r'^\s*(?:class|interface)\s+\w+', code_snippet, flags=re.MULTILINE))
    
    unique_words = set(words_in_code)
    features_dict['unique_word_ratio_no_comments'] = len(unique_words) / len(words_in_code) if words_in_code else 0.0

    all_tokens = re.findall(r'\b\w+\b|[^\w\s]', code_no_comments_and_strings)
    features_dict['token_count'] = len(all_tokens)
    features_dict['avg_token_length'] = np.mean([len(token) for token in all_tokens]) if all_tokens else 0.0
    features_dict['keyword_density'] = features_dict['keyword_count'] / features_dict['token_count'] if features_dict['token_count'] > 0 else 0.0
    features_dict['operator_density'] = features_dict['operator_count'] / features_dict['token_count'] if features_dict['token_count'] > 0 else 0.0
    features_dict['punctuator_count'] = sum(1 for char in code_no_comments_and_strings if char in '.,;(){}[]:<>!@#$%^&*-+/?|\\`~')

    max_indent_depth = 0
    for line in lines_of_code:
        leading_spaces = len(line) - len(line.lstrip(' '))
        leading_tabs = len(line) - len(line.lstrip('\t'))
        max_indent_depth = max(max_indent_depth, leading_spaces, leading_tabs)
    features_dict['max_indent_depth'] = max_indent_depth

    param_counts = []
    function_matches = re.findall(r'(?:def|function|void|int|string)\s+\w+\s*\((.*?)\):?', code_no_comments_and_strings, re.DOTALL | re.IGNORECASE)
    for match in function_matches:
        params = [p.strip() for p in match.split(',') if p.strip()]
        param_counts.append(len(params))
    features_dict['avg_params_per_function'] = np.mean(param_counts) if param_counts else 0.0

    non_keyword_words = [word for word in words_in_code if word.lower() not in common_keywords]
    features_dict['halstead_unique_operands'] = len(set(non_keyword_words))
    features_dict['halstead_total_operands'] = len(non_keyword_words)

    cyclomatic_keywords = ['if', 'for', 'while', 'elif', 'else', 'case', 'default', 'try', 'except', 'finally', 'and', 'or', 'switch']
    features_dict['cyclomatic_complexity_proxy'] = sum(code_no_comments_and_strings.lower().count(kw) for kw in cyclomatic_keywords) + 1 # +1 for the function entry point

    identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code_no_comments_and_strings)
    features_dict['avg_identifier_length'] = np.mean([len(id) for id in identifiers]) if identifiers else 0.0

    features_dict['unique_operators'] = len(set(re.findall(r'[+\-*/%=&|^~<>!]+', code_no_comments_and_strings)))

    short_lines = sum(1 for line in lines_of_code if len(line) < 40)
    features_dict['ratio_short_lines'] = short_lines / num_lines_no_comments if num_lines_no_comments > 0 else 0.0

    features_dict['combined_literal_count'] = features_dict['numeric_literal_count'] + features_dict['string_literal_count']

    current_depth = 0
    total_depth = 0
    depth_counts = 0
    for line in lines_of_code:
        if line and isinstance(line, str) and ('{' in line or line.strip().endswith(':')):
            current_depth += 1
        if '}' in line:
            current_depth = max(0, current_depth - 1)
        total_depth += current_depth
        depth_counts += 1
    features_dict['avg_nested_block_depth'] = total_depth / depth_counts if depth_counts > 0 else 0.0

    features_dict['func_param_variance'] = np.var(param_counts) if len(param_counts) > 1 else 0.0

    long_lines = sum(1 for line in lines_of_code if len(line) > 80)
    features_dict['ratio_long_lines'] = long_lines / num_lines_no_comments if num_lines_no_comments > 0 else 0.0

    max_paren_depth = 0
    current_paren_depth = 0
    for char in code_no_comments_and_strings:
        if char in '([{':
            current_paren_depth += 1
            max_paren_depth = max(max_paren_depth, current_paren_depth)
        elif char in ')]}':
            current_paren_depth = max(0, current_paren_depth - 1)
    features_dict['max_paren_depth'] = max_paren_depth

    distinct_types = set(re.findall(r'\b(?:int|float|double|char|bool|string|list|dict|set|tuple|array|void)\b', code_no_comments_and_strings.lower()))
    features_dict['num_distinct_data_types'] = len(distinct_types)

    lines_per_function = [len([l for l in f.split('\n') if l.strip()]) for f in function_matches]
    features_dict['avg_lines_per_function'] = np.mean(lines_per_function) if lines_per_function else 0.0

    features_dict['use_of_fstrings_template_literals'] = 1.0 if len(re.findall(r'f".*?"|f\'.*?\'|`.*?`', code_snippet)) > 0 else 0.0

    features_dict['num_try_except_blocks'] = code_no_comments_and_strings.lower().count('try:') + code_no_comments_and_strings.lower().count('try {')

    magic_numbers = len(re.findall(r'(?<![a-zA-Z_0-9])\b\d+\b(?![a-zA-Z_0-9])', code_no_comments_and_strings))
    total_numbers = features_dict['numeric_literal_count']
    features_dict['ratio_magic_numbers'] = magic_numbers / total_numbers if total_numbers > 0 else 0.0

    features_dict['has_tf_keras_boilerplate'] = 1.0 if (
        'sequential' in code_snippet.lower() and
        'dense' in code_snippet.lower() and
        'flatten' in code_snippet.lower() and
        'compile' in code_snippet.lower() and
        'fit' in code_snippet.lower()
    ) else 0.0

    # New Complexity Features
    features_dict['num_logical_operators'] = sum(code_no_comments_and_strings.lower().count(op) for op in [' and ', ' or ', '&&', '||'])
    features_dict['num_comparison_operators'] = sum(code_no_comments_and_strings.lower().count(op) for op in ['==', '!=', '<', '>', '<=', '>='])
    
    # Heuristic for ratio of nested conditionals (simple count of 'if' followed by 'if' or 'else' on next line)
    nested_if_count = 0
    for i in range(len(lines_of_code) - 1):
        if 'if' in lines_of_code[i].lower():
            next_line_stripped = lines_of_code[i+1].strip().lower()
            if next_line_stripped.startswith('if') or next_line_stripped.startswith('else if') or next_line_stripped.startswith('elif'):
                nested_if_count += 1
    features_dict['ratio_nested_conditionals'] = nested_if_count / features_dict['conditional_count'] if features_dict['conditional_count'] > 0 else 0.0


    return features_dict

# --- 3. Synthetic Data Generation Function (for AI code only) ---
# Removed live API calls for synthetic data generation to avoid 403/429 errors and speed up execution.
# This function is now a placeholder.
async def generate_synthetic_code(prompt, num_samples=5):
    print(f"Skipping live API generation for prompt: '{prompt}'. Using hardcoded data.")
    return [] # Return empty list as synthetic data is now hardcoded in prepare_combined_dataset

# --- Function to fetch and parse CSV ---
async def fetch_and_parse_csv(url):
    print(f"Fetching data from: {url}")
    retries = 3
    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                # First attempt to get the raw content directly if it's a raw URL
                raw_url = url.replace("/blob/", "/raw/")
                
                try:
                    async with session.get(raw_url) as response:
                        response.raise_for_status() # Raise an exception for HTTP errors
                        csv_content = await response.text()
                        # Check if it's actual CSV or an HTML error page
                        if "<!DOCTYPE html>" in csv_content.lower() or "<html" in csv_content.lower():
                            raise ValueError("Raw URL returned HTML, trying original URL.")
                except (aiohttp.client_exceptions.ClientError, ValueError):
                    # If raw URL fails or returns HTML, try the original blob URL
                    async with session.get(url) as response:
                        response.raise_for_status() # Raise an exception for HTTP errors
                        csv_content = await response.text()
                        # For GitHub 'blob' links, the content is often embedded in HTML.
                        # We need to extract the raw text if it's an HTML page.
                        if "<!DOCTYPE html>" in csv_content.lower():
                            # A simple heuristic: try to find pre-formatted code blocks or similar
                            # This is a very basic attempt and might need more robust parsing for complex HTML
                            match = re.search(r'<pre[^>]*>(.*?)</pre>', csv_content, re.DOTALL | re.IGNORECASE)
                            if match:
                                csv_content = match.group(1)
                            else:
                                # If no pre tag, assume the content itself is the CSV if it looks like one
                                # This is a weak assumption, but better than nothing if no raw link works
                                pass # csv_content already holds the response text

            # Use pandas to read the CSV content from a string
            df = pd.read_csv(io.StringIO(csv_content))
            print(f"Successfully fetched and parsed CSV from {url}. Shape: {df.shape}")
            return df
        except aiohttp.client_exceptions.ClientError as e:
            print(f"Network or HTTP error fetching CSV from {url} (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt) # Exponential backoff
            else:
                print(f"Max retries reached for CSV: {url}. Skipping.")
                return pd.DataFrame() # Return empty DataFrame if all retries fail
        except Exception as e:
            print(f"Error fetching or parsing CSV from {url}: {e}")
            return pd.DataFrame() # Return empty DataFrame on other errors

# --- 4. Prepare Dataset from CSVs and Synthetic AI Data ---
async def prepare_combined_dataset():
    all_code_snippets = []
    all_labels = []

    github_csv_urls = [
        "https://github.com/Back3474/AI-Human-Generated-Program-Code-Dataset/blob/main/AI-Human-Generated-Program-Code-Dataset(1).csv",
        "https://github.com/Back3474/AI-Human-Generated-Program-Code-Dataset/blob/main/AI-Human-Generated-Program-Code-Dataset(2).csv"
    ]

    for url in github_csv_urls:
        df = await fetch_and_parse_csv(url)
        if not df.empty:
            # Based on previous browsing output, these columns are expected
            if 'ai_generated_code' in df.columns and 'human_generated_code' in df.columns:
                for index, row in df.iterrows():
                    ai_code = row['ai_generated_code']
                    human_code = row['human_generated_code']
                    if pd.notna(ai_code) and str(ai_code).strip():
                        all_code_snippets.append(str(ai_code))
                        all_labels.append("AI")
                    if pd.notna(human_code) and str(human_code).strip():
                        all_code_snippets.append(str(human_code))
                        all_labels.append("Human")
            elif 'code' in df.columns and 'label' in df.columns: # Fallback for a more standard format
                for index, row in df.iterrows():
                    code = row['code']
                    label = row['label']
                    if pd.notna(code) and str(code).strip():
                        all_code_snippets.append(str(code))
                        all_labels.append(str(label))
            else:
                print(f"Warning: Columns 'ai_generated_code'/'human_generated_code' or 'code'/'label' not found in {url}. Please check CSV structure.")

    print(f"Total snippets from GitHub CSVs: {len(all_code_snippets)}")

    # Hardcoded synthetic AI code snippets (expanded for diversity)
    # This replaces dynamic API generation to avoid network issues and rate limits.
    synthetic_ai_code_snippets = [ # Initialize here
        """
import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# This is a basic sigmoid function implementation.
""",
        """
function factorial(n) {
  if (n === 0) {
    return 1;
  } else {
    return n * factorial(n - 1);
  }
}
// Recursive factorial function in JavaScript.
""",
        """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, Java World!");
    }
}
// Standard Java Hello World program.
""",
        """
#include <iostream>
int main() {
    std::cout << "Hello, C++!" << std::endl;
    return 0;
}
// Basic C++ program.
""",
        """
CREATE TABLE Users (
    UserID INT PRIMARY KEY,
    UserName VARCHAR(50) NOT NULL
);
-- SQL to create a simple Users table.
""",
        """
<!DOCTYPE html>
<html>
<head>
<title>AI Generated Page</title>
<style>
  body { font-family: Arial, sans-serif; }
  h1 { color: #333; }
</style>
</head>
<body>
  <h1>Welcome to my AI Page</h1>
  <p>This content was generated by an AI model.</p>
</body>
</html>
""",
        """
.container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background-color: #f0f0f0;
}
/* Basic CSS for centering a div. */
""",
        """
package main
import "fmt"
func main() {
    fmt.Println("Hello, Go!")
}
// Go language Hello World.
""",
        """
puts "Hello, Ruby World!"
# Simple Ruby print statement.
""",
        """
using System;
public class Product
{
    public int Id { get; set; }
    public string Name { get; set; }
}
// C# Product class.
""",
        """
object HelloWorld {
  def main(args: Array[String]): Unit = {
    println("Hello, Scala!")
  }
}
// Scala Hello World object.
""",
        """
func greet(name: String) -> String {
    return "Hello, " + name + "!"
}
// Swift greeting function.
""",
        """
#!/bin/bash
echo "Listing files in current directory:"
ls -l
# Simple shell script.
""",
        """
import React from 'react';
function MyButton() {
  return (
    <button onClick={() => alert('Button clicked!')}>
      Click Me
    </button>
  );
}
export default MyButton;
// Basic React functional component.
""",
        """
# This is a comment only file.
# It contains no executable code.
# Just some notes.
"""
    ]
    # No longer calling generate_synthetic_code from API, directly using hardcoded list

    print(f"Using {len(synthetic_ai_code_snippets)} hardcoded synthetic AI snippets.")


    # Combining all data
    combined_code_snippets = all_code_snippets + synthetic_ai_code_snippets
    combined_labels = all_labels + ["AI"] * len(synthetic_ai_code_snippets)

    if not combined_code_snippets:
        print("No data available for training after combining CSVs and synthetic AI. Exiting.")
        return np.array([]), np.array([])

    X_combined = []
    y_combined = []

    for i, code in enumerate(combined_code_snippets):
        features_dict = extract_features(code)
        # Convert feature dictionary to ordered numpy array
        feature_vector = np.array([features_dict[key] for key in FEATURE_ORDER])
        X_combined.append(feature_vector)
        y_combined.append(combined_labels[i])

    X_combined = np.array(X_combined)
    y_combined = np.array(y_combined)

    print(f"Final combined dataset size: {len(y_combined)} samples.")
    return X_combined, y_combined

# --- 5. Retrain the Model (for Code Detection, from scratch) ---
async def retrain_code_detection_model_from_scratch():
    print("Preparing combined dataset from GitHub CSVs and synthetic AI data for code detection...")
    X_data, y_data = await prepare_combined_dataset()

    if X_data.size == 0 or len(np.unique(y_data)) < 2:
        print("Insufficient data or labels to train the code detection model. Returning default models.")
        return MLPClassifier(), StandardScaler(), LabelEncoder()

    # Initialize new scaler, label encoder, and neural model
    new_label_encoder = LabelEncoder()
    new_scaler = StandardScaler()
    
    # Re-initialize MLPClassifier with robust parameters for better performance
    new_neural_model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), # Reduced complexity for faster Canvas execution
        max_iter=500, # Reduced iterations
        activation='relu',
        solver='adam',
        random_state=42,
        alpha=0.0001, # L2 regularization
        learning_rate='adaptive', # Adaptive learning rate
        early_stopping=True, # Enable early stopping to prevent overfitting
        validation_fraction=0.15, # Use 15% of training data for validation
        n_iter_no_change=20, # Reduced patience for early stopping
        verbose=False # Set to False for cleaner output and slightly faster execution
    )

    # Fit label encoder on all unique labels
    new_label_encoder.fit(y_data)
    y_encoded = new_label_encoder.transform(y_data)

    # Fit scaler on the features
    new_scaler.fit(X_data)
    X_scaled = new_scaler.transform(X_data)

    # Split data for training and validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training data size for code detection: {len(X_train)}")
    print(f"Test data size for code detection: {len(X_test)}")
    print("Training the new neural network model from scratch for code detection...")
    new_neural_model.fit(X_train, y_train)

    # Evaluate the newly trained model
    y_pred = new_neural_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=new_label_encoder.classes_)

    print(f"\nNew code detection model training complete. Accuracy: {accuracy:.4f}")
    print("Code Detection Classification Report:\n", report)

    return new_neural_model, new_scaler, new_label_encoder

# --- 6. Prediction Function (for Code Detection) ---
def predict_code_type(code, model, scaler, label_encoder, explain=False): # Changed vectorizer to scaler
    """
    Predicts whether the code snippet is AI-generated or human-generated,
    with nuanced labels based on probabilities and heuristics, and feature insights.
    Returns a dictionary with predicted_type, probabilities, nuanced_label, and feature_insights.
    """
    if not isinstance(code, str) or not code.strip():
        return {
            "predicted_type": "Invalid Input",
            "probabilities": {"AI": 0.5, "Human": 0.5},
            "nuanced_label": "Invalid Input",
            "feature_insights": {}
        }

    # Step 1: Extract features from code
    features_dict = extract_features(code)
    features_vector = np.array([features_dict[key] for key in FEATURE_ORDER]) # Convert dict to ordered array

    expected_features_count = scaler.n_features_in_
    if len(features_vector) != expected_features_count:
        print(f"Warning: Feature vector size mismatch. Expected {expected_features_count}, got {len(features_vector)}. Padding/truncating.")
        if len(features_vector) < expected_features_count:
            features_vector = np.pad(features_vector, (0, expected_features_count - len(features_vector)), 'constant')
        else:
            features_vector = features_vector[:expected_features_count]

    features_scaled = scaler.transform(features_vector.reshape(1, -1))

    # Step 2: Predict probability
    proba = model.predict_proba(features_scaled)[0]
    classes = label_encoder.classes_ # Use label_encoder.classes_ for class names

    # Build probability dictionary
    proba_dict = {cls: float(prob) for cls, prob in zip(classes, proba)}

    # Cap probabilities to avoid overconfidence
    proba_dict = {
        cls: min(max(prob, MIN_OUTPUT_PROB), MAX_OUTPUT_PROB)
        for cls, prob in proba_dict.items()
    }

    # Normalize after capping
    total = sum(proba_dict.values())
    if total > 0: # Avoid division by zero
        proba_dict = {cls: prob / total for cls, prob in proba_dict.items()}
    else: # Fallback if total is zero (should not happen with MIN_OUTPUT_PROB)
        proba_dict = {"AI": 0.5, "Human": 0.5}


    ai_prob = proba_dict.get("AI", 0.0)
    human_prob = proba_dict.get("Human", 0.0)

    # Step 3: Adjust confidence for trivial/simple code
    is_simple_code = (
        features_dict['num_lines_no_comments'] < 8 and
        features_dict['function_defs'] <= 1 and
        features_dict['loop_count'] <= 1 and
        features_dict['operator_count'] < 10 and
        features_dict['combined_literal_count'] < 5
    )

    if is_simple_code:
        ai_prob = (ai_prob * 0.7) + 0.15
        human_prob = (human_prob * 0.7) + 0.15
        total = ai_prob + human_prob
        if total > 0:
            ai_prob /= total
            human_prob /= total
        else:
            ai_prob = 0.5
            human_prob = 0.5


    # Step 4: Reduce certainty if feature diversity is low
    # Convert sparse feature vector to dense array for sum check
    nonzero_feature_count = np.sum(features_vector != 0) # Count non-zero features in the raw vector
    if nonzero_feature_count < 10: # If less than 10 features are non-zero
        ai_prob = (ai_prob * 0.8) + 0.1
        human_prob = (human_prob * 0.8) + 0.1
        total = ai_prob + human_prob
        if total > 0:
            ai_prob /= total
            human_prob /= total
        else:
            ai_prob = 0.5
            human_prob = 0.5

    # Update probability dictionary again after adjustments
    proba_dict['AI'] = ai_prob
    proba_dict['Human'] = human_prob

    # Step 5: Assign label based on confidence thresholds
    confidence_threshold_high = 0.85
    confidence_threshold_medium = 0.60
    confidence_threshold_undetectable_range = 0.15 # If abs(AI-Human) prob < this, consider undetectable/ambiguous

    # Default to Ambiguous/Mixed
    label = "Ambiguous/Mixed"

    # Check for strong, definitive AI patterns first, overriding all else
    if features_dict['has_tf_keras_boilerplate'] == 1.0:
        label = "AI Generated" # Force to AI if TF/Keras boilerplate is detected
    elif ai_prob >= confidence_threshold_high:
        label = "AI Generated"
    elif human_prob >= confidence_threshold_high:
        label = "Human Generated"
    elif abs(ai_prob - human_prob) < confidence_threshold_undetectable_range:
        # If probabilities are very close, and no strong AI pattern (like TF boilerplate)
        # then classify as undetectable or ambiguous
        if ai_prob > human_prob:
            label = "Undetectable AI"
        elif human_prob > ai_prob:
            label = "Undetectable Human"
        else: # Exactly 50/50
            label = "Ambiguous/Mixed"
    elif ai_prob >= confidence_threshold_medium:
        label = "AI Refined Human" # Predicted AI, but Human prob is still substantial
    elif human_prob >= confidence_threshold_medium:
        label = "Human Refined AI" # Predicted Human, but AI prob is still substantial
    else:
        label = "Ambiguous/Mixed" # Fallback for other less confident cases

    result = {
        "predicted_type": "AI" if ai_prob > human_prob else "Human", # Based on final adjusted probs
        "probabilities": proba_dict,
        "nuanced_label": label, # Use the new 'label' as nuanced_label
        "feature_insights": {} # Initialize feature_insights here
    }

    # --- Feature Insights / Ratios on Various Fields (Re-calculated based on features_dict) ---
    style_score = 0
    if features_dict['indent_consistency'] == 1.0: style_score += 2
    if features_dict['blank_lines_ratio'] > 0.1: style_score += 1 # Some blank lines for readability
    if features_dict['avg_identifier_length'] > 5: style_score += 1 # Longer, descriptive identifiers
    if features_dict['ratio_short_lines'] > 0.3: style_score += 1 # Many short lines for readability
    
    if style_score >= 4: result['feature_insights']['Code Style Rate'] = "Excellent (Human-like)"
    elif style_score >= 2: result['feature_insights']['Code Style Rate'] = "Good (Mixed/Standard)"
    else: result['feature_insights']['Code Style Rate'] = "Poor (AI-like/Compact)"

    complexity_score = 0
    # Incorporate new complexity features
    if features_dict['cyclomatic_complexity_proxy'] > 15: complexity_score += 2 # High complexity
    if features_dict['avg_lines_per_function'] > 25: complexity_score += 1 # Longer functions
    if features_dict['num_try_except_blocks'] > 0: complexity_score += 1 # Error handling
    if features_dict['num_logical_operators'] > 5: complexity_score += 1 # More logical conditions
    if features_dict['num_comparison_operators'] > 10: complexity_score += 1 # More comparisons
    if features_dict['ratio_nested_conditionals'] > 0.3: complexity_score += 1 # Significant nesting
    
    if complexity_score >= 4: result['feature_insights']['Complexity Rate'] = "Very High (Complex Human/AI)"
    elif complexity_score >= 2: result['feature_insights']['Complexity Rate'] = "High (Human/Complex AI)"
    elif complexity_score >= 1: result['feature_insights']['Complexity Rate'] = "Medium (Standard)"
    else: result['feature_insights']['Complexity Rate'] = "Low (Simple AI/Boilerplate)"

    boilerplate_score = 0
    if features_dict['num_imports'] > 5: boilerplate_score += 1
    if features_dict['num_classes'] > 1: boilerplate_score += 1
    if features_dict['function_defs'] > 5: boilerplate_score += 1
    
    if boilerplate_score >= 2: result['feature_insights']['Boilerplate Tendency'] = "High (AI-like)"
    else: result['feature_insights']['Boilerplate Tendency'] = "Low (Human-like/Minimal)"

    readability_score = 0
    if features_dict['ratio_short_lines'] > 0.4: readability_score += 1
    if features_dict['ratio_long_lines'] < 0.1: readability_score += 1
    if features_dict['original_comment_ratio'] > 0.1: readability_score += 1
    
    if readability_score >= 2: result['feature_insights']['Readability'] = "High (Human-like)"
    else: result['feature_insights']['Readability'] = "Low (AI-like/Dense)"

    if features_dict['has_tf_keras_boilerplate'] == 1.0:
        result['feature_insights']['Specific AI Pattern'] = "TensorFlow/Keras Boilerplate Detected (Strong AI Indicator)"
    else:
        result['feature_insights']['Specific AI Pattern'] = "No Specific AI Framework Boilerplate Detected"

    if explain:
        result["features_raw"] = features_dict # Include raw features if explain is True

    return result

# --- Main Execution ---
async def main():
    global neural_model, scaler, label_encoder # Allow modification of global variables

    # --- Train/Load Code Detection Model ---
    # Load and evaluate existing 'simple' model
    simple_model, simple_scaler, simple_label_encoder = load_existing_model_components('simple_neural')
    # Load and evaluate existing 'improved' model
    improved_model, improved_scaler, improved_label_encoder = load_existing_model_components('improved')

    retrained_model, updated_scaler, updated_label_encoder = await retrain_code_detection_model_from_scratch()
    neural_model = retrained_model
    scaler = updated_scaler
    label_encoder = updated_label_encoder

    print("\n--- Testing Newly Trained Code Detection Model with Example Snippets ---")

    # Example 1: Human-like Python snippet
    human_code_example = """
def calculate_sum(a, b):
    # This function adds two numbers
    result = a + b
    return result # Returns the sum
"""
    print(f"\nSnippet 1 (Human-like):\n{human_code_example}")
    prediction_result = predict_code_type(human_code_example, neural_model, scaler, label_encoder)
    print(f"Predicted type: {prediction_result['predicted_type']}")
    print(f"Probabilities: {prediction_result['probabilities']['AI']:.4f} AI, {prediction_result['probabilities']['Human']:.4f} Human") # Formatted output
    print(f"Nuanced Label: {prediction_result['nuanced_label']}")
    print(f"Feature Insights: {prediction_result['feature_insights']}")

    # Example 2: AI-like Python snippet (more structured, typical AI output)
    ai_code_example = """
import math

def factorial(n: int) -> int:
    \"\"\"
    Calculates the factorial of a non-negative integer.
    This function computes n! using an iterative approach.
    \"\"\"
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    elif n == 0:
        return 1
    else:
        res = 1
        for i in range(1, n + 1):
            res *= i
        return res

# Example usage:
# num = 5
# print(f"The factorial of {num} is {factorial(num)}")
"""
    print(f"\nSnippet 2 (AI-like):\n{ai_code_example}")
    prediction_result = predict_code_type(ai_code_example, neural_model, scaler, label_encoder)
    print(f"Predicted type: {prediction_result['predicted_type']}")
    print(f"Probabilities: {prediction_result['probabilities']['AI']:.4f} AI, {prediction_result['probabilities']['Human']:.4f} Human") # Formatted output
    print(f"Nuanced Label: {prediction_result['nuanced_label']}")
    print(f"Feature Insights: {prediction_result['feature_insights']}")

    # Example 7: User's problematic AI-generated TensorFlow snippet
    tf_ai_snippet = """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")
"""
    print(f"\nSnippet 7 (User's AI TensorFlow):\n{tf_ai_snippet}")
    prediction_result = predict_code_type(tf_ai_snippet, neural_model, scaler, label_encoder)
    print(f"Predicted type: {prediction_result['predicted_type']}")
    print(f"Probabilities: {prediction_result['probabilities']['AI']:.4f} AI, {prediction_result['probabilities']['Human']:.4f} Human") # Formatted output
    print(f"Nuanced Label: {prediction_result['nuanced_label']}")
    print(f"Feature Insights: {prediction_result['feature_insights']}")


    # You can save the newly trained code detection model if needed
    print("\nSaving newly trained code detection models as .pkl files...")
    with open('newly_trained_neural_model.pkl', 'wb') as f:
        pickle.dump(neural_model, f)
    with open('newly_trained_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('newly_trained_label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("Models saved successfully.")

    # --- Accuracy Comparison ---
    print("\n--- Model Accuracy Comparison ---")
    # We will use the test set from the newly trained model for comparison
    # This is to ensure all models are evaluated on the same data for a fair comparison.
    X_compare_raw, y_compare_raw = await prepare_combined_dataset() # Re-prepare data to get X_data, y_data
    
    # Convert raw features to ordered numpy arrays for scaling
    X_compare_features_dicts = [extract_features(code_snippet) for code_snippet in X_compare_raw]
    X_compare_features_vectors = np.array([features_dict[key] for features_dict in X_compare_features_dicts for key in FEATURE_ORDER]).reshape(len(X_compare_features_dicts), -1)
    
    X_compare_scaled = scaler.transform(X_compare_features_vectors)
    y_compare_encoded = label_encoder.transform(y_compare_raw)
    
    # Split for a consistent test set for all models
    _, X_test_compare, _, y_test_compare = train_test_split(
        X_compare_scaled, y_compare_encoded, test_size=0.2, random_state=42, stratify=y_compare_encoded
    )

    if simple_model and simple_scaler and simple_label_encoder:
        try:
            # Need to transform X_test_compare with simple_scaler (if feature sizes match)
            # This is a critical point: if feature extraction changed, simple_scaler might not fit.
            # For this comparison, we assume the simple_scaler can transform the current features.
            # If the feature dimensions don't match, this will fail.
            if simple_scaler.n_features_in_ == X_test_compare.shape[1]:
                X_test_simple_scaled = simple_scaler.transform(X_test_compare)
                simple_acc = accuracy_score(y_test_compare, simple_model.predict(X_test_simple_scaled))
                print(f"Simple Model Accuracy: {simple_acc:.4f} ({simple_acc*100:.2f}%)")
            else:
                print(f"Simple Model Accuracy: N/A (Feature dimension mismatch: Expected {simple_scaler.n_features_in_}, got {X_test_compare.shape[1]})")
        except Exception as e:
            print(f"Could not evaluate Simple Model due to error: {e}")
            print("Simple Model Accuracy: N/A")
    else:
        print("Simple Model Accuracy: N/A (Model not loaded)")

    if improved_model and improved_scaler and improved_label_encoder:
        try:
            # Similar assumption for improved_scaler
            if improved_scaler.n_features_in_ == X_test_compare.shape[1]:
                X_test_improved_scaled = improved_scaler.transform(X_test_compare)
                improved_acc = accuracy_score(y_test_compare, improved_model.predict(X_test_improved_scaled))
                print(f"Improved Model Accuracy: {improved_acc:.4f} ({improved_acc*100:.2f}%)")
            else:
                print(f"Improved Model Accuracy: N/A (Feature dimension mismatch: Expected {improved_scaler.n_features_in_}, got {X_test_compare.shape[1]})")
        except Exception as e:
            print(f"Could not evaluate Improved Model due to error: {e}")
            print("Improved Model Accuracy: N/A")
    else:
        print("Improved Model Accuracy: N/A (Model not loaded)")
    
    # Current model accuracy (already calculated during training)
    current_model_acc = accuracy_score(y_test_compare, neural_model.predict(X_test_compare))
    print(f"Current Model Accuracy: {current_model_acc:.4f} ({current_model_acc*100:.2f}%)")


# Run the main asynchronous function
import asyncio
asyncio.run(main())
