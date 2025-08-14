# Import the necessary libraries
import os
import google.generativeai as genai

# Configure the API key from the environment variable
# The library will automatically find and use the API_KEY environment variable.
genai.configure(api_key=os.environ["API_KEY"])

# Now you can use the Gemini model
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Hello, how can I help you today?")
print(response.text)

