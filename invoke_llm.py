# Program to invoke an LLM

import os
import google.generativeai as genai

try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    exit()

# Create the model instance
# For text-only prompts, use 'gemini-2.0-flash'
model = genai.GenerativeModel('gemini-2.0-flash')

# The prompt you want to send to Gemini
prompt = "Write a limerick python programming language"

print(f"Sending prompt: {prompt}\n")

# Generate content
try:
    response = model.generate_content(prompt)
    
    # Print the response text
    print("--- Gemini's Response ---")
    print(response.text)
    print("-------------------------")

except Exception as e:
    print(f"An error occurred: {e}")    
