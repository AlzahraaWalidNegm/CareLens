import os
from fastapi import FastAPI, HTTPException, Depends, Header
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Load API key from .env file
API_KEY = os.getenv("API_KEY", "default_secret_key")  # Fallback key

def verify_api_key(api_key: str = Header(None)):
    """Verify the API key provided in headers."""
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.get("/")
def home():
    return {"message": "Welcome to FastAPI!"}

@app.get("/secure-endpoint")
def secure_endpoint(api_key: str = Depends(verify_api_key)):
    return {"message": "You have access to this secure endpoint!"}