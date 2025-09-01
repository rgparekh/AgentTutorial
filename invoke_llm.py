# Program to invoke an LLM

import os
from google import genai
from google.genai import types


try:
    google_api_key = os.environ["GOOGLE_API_KEY"]
    print("Google API Key loaded successfully.")
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set.")
    google_api_key = None # Or handle the error as appropriate

try:
    client = genai.Client(
        api_key=google_api_key
    )    

    prompt = 'Explain quantum physics to a 10-year old in 200 words or less'

    print(f"Sending prompt: {prompt}\n")

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt
    )

    # Print the response text
    print("--- Gemini's Response ---")
    print(response.text)
    print("-------------------------")

except Exception as e:
    print(f"An error occurred: {e}")    
    
