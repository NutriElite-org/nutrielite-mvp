#!/usr/bin/env python3
"""
Debug script to test JSON cleaning on your model's actual output
"""

import json
import re

def clean_json_string(text):
    """Clean and fix common JSON formatting issues"""
    # Remove comments (// style)
    text = re.sub(r'//.*?(?=\n|$)', '', text)
    
    # Remove JavaScript-style comments (/* */)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    
    # Fix common JSON issues
    text = text.replace("'", '"')  # Replace single quotes with double quotes
    text = re.sub(r',\s*}', '}', text)  # Remove trailing commas before }
    text = re.sub(r',\s*]', ']', text)  # Remove trailing commas before ]
    
    # Fix missing quotes around property names
    text = re.sub(r'(\w+)(?=\s*:)', r'"\1"', text)
    
    # Fix unquoted string values that should be quoted
    text = re.sub(r':\s*([a-zA-Z][a-zA-Z0-9_\s]*?)(?=\s*[,}\]])', r': "\1"', text)
    
    # Remove extra whitespace and newlines within JSON
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_json_from_response(response_text):
    """Extract and clean JSON from model response"""
    # Try different approaches to find JSON
    
    # Method 1: Look for complete JSON object
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, response_text, re.DOTALL)
    
    if matches:
        # Take the longest match (most likely to be complete)
        json_candidate = max(matches, key=len)
        return clean_json_string(json_candidate)
    
    # Method 2: Find first { to last }
    start = response_text.find('{')
    end = response_text.rfind('}')
    
    if start != -1 and end != -1 and end > start:
        json_candidate = response_text[start:end+1]
        return clean_json_string(json_candidate)
    
    raise ValueError("No valid JSON structure found in response")

# Test with your actual model output
test_response = '''3560,
"_2g_protein_per_kg": 170, // Protein requirement based on 2g per kg body weight
"%_Carbohydrates": 55,
"% fat" : 25짜
},
" meal plan": [
"time": "Breakfast",
"_type": "Meal"
"items": [ "Egg white omelet (6 whites); Spinach (2 cups); Feta cheese (1 oz); Whole grain toast (1 slice)" ], nutrition_": { "calcium mg": 420, "vitamin_a_iu": 9800, ... }
},
..
"_suppLEMENTS": [ // Include only supplements that are certified by relevant spor...'''

print("Original response:")
print(test_response)
print("\n" + "="*50 + "\n")

try:
    cleaned = extract_json_from_response(test_response)
    print("Cleaned JSON:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Additional specific cleaning for your model's patterns
    cleaned = re.sub(r'"_(\w+)":', r'"\1":', cleaned)  # Fix _property names
    cleaned = re.sub(r'"%([^"]+)":', r'"\1":', cleaned)  # Fix %property names
    cleaned = re.sub(r':\s*(\d+)짜', r': \1', cleaned)  # Remove Korean character
    cleaned = re.sub(r'\[\s*"([^"]+)"\s*\]\s*,', r'["\1"],', cleaned)  # Fix array formatting
    
    print("Final cleaned JSON:")
    print(cleaned)
    print("\n" + "="*50 + "\n")
    
    # Try to parse
    parsed = json.loads(cleaned)
    print("SUCCESS! Parsed JSON:")
    print(json.dumps(parsed, indent=2))
    
except Exception as e:
    print(f"ERROR: {e}")
    print("You may need to adjust the cleaning patterns for your specific model output")
