# Simple examples for getting structured output from Google Gemini API

import os
import json
import google.generativeai as genai

# Configure the API
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit()

model = genai.GenerativeModel('gemini-2.0-flash')

def get_json_output():
    """Get structured JSON output from Gemini"""
    print("=== JSON Structured Output ===")
    
    prompt = """
    Create a Python function documentation in JSON format with the following structure:
    {
        "function_name": "name",
        "description": "what it does",
        "parameters": [
            {"name": "param1", "type": "str", "description": "param description"}
        ],
        "return_value": {"type": "str", "description": "return description"},
        "example": "code example"
    }
    
    Document a function that calculates the factorial of a number.
    Return ONLY valid JSON, no additional text.
    """
    
    try:
        response = model.generate_content(prompt)
        print("Raw response:")
        print(response.text)
        
        # Parse the JSON
        parsed = json.loads(response.text)
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2))
        
        return parsed
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_formatted_output():
    """Get structured output with specific formatting"""
    print("\n=== Formatted Structured Output ===")
    
    prompt = """
    Write a Python code review in this exact format:
    
    TITLE: [Title here]
    
    SUMMARY: [Summary here]
    
    ISSUES:
    - [Issue 1]
    - [Issue 2]
    
    SUGGESTIONS:
    - [Suggestion 1]
    - [Suggestion 2]
    
    RATING: [1-10]
    
    Review this code: def add(a, b): return a + b
    """
    
    try:
        response = model.generate_content(prompt)
        print(response.text)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_markdown_output():
    """Get structured output in Markdown format"""
    print("\n=== Markdown Structured Output ===")
    
    prompt = """
    Create a Python tutorial in this exact markdown format:
    
    # [Title]
    
    ## Overview
    [Overview text]
    
    ## Code Example
    ```python
    [code here]
    ```
    
    ## Explanation
    [Explanation text]
    
    ## Best Practices
    - [Practice 1]
    - [Practice 2]
    
    Write about Python list comprehensions.
    """
    
    try:
        response = model.generate_content(prompt)
        print(response.text)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_csv_output():
    """Get structured output in CSV format"""
    print("\n=== CSV Structured Output ===")
    
    prompt = """
    Create a list of 3 Python data structures in CSV format:
    Name,Description,Use Case,Complexity
    
    Example:
    List,Ordered collection of items,Storing sequences of data,Low
    """
    
    try:
        response = model.generate_content(prompt)
        print(response.text)
        return response.text
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("Getting Structured Output from Gemini API\n")
    
    # Run examples
    get_json_output()
    get_formatted_output()
    get_markdown_output()
    get_csv_output() 