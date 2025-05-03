from pyngrok import ngrok
import os
import subprocess
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure ngrok (token is loaded from .env file)
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

# Start the Streamlit app in the background
streamlit_process = subprocess.Popen(["streamlit", "run", "ll_parser_app.py"])

# Wait for Streamlit to start
print("Waiting for Streamlit to start...")
time.sleep(5)  # Allow Streamlit time to initialize

# Connect ngrok to the Streamlit port (default is 8501)
public_url = ngrok.connect(8501)
print(f"Public URL: {public_url}")

try:
    # Keep the script running
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    # Handle Ctrl+C
    print("Closing ngrok tunnel and Streamlit...")
    ngrok.disconnect(public_url)
    streamlit_process.terminate()
    print("Done!")
