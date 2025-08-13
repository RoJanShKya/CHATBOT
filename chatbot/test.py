import requests
import os
from dotenv import load_dotenv

load_dotenv()

def verify_github_token():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return "❌ No token found in .env"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }
    
    try:
        response = requests.get(
            "https://api.github.com/repos/RoJanShKya/ChatPdf",  # Replace with your repo
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            return "✅ Token works! Repository accessible"
        return f"❌ Token failed (HTTP {response.status_code}): {response.json().get('message')}"
    except Exception as e:
        return f"🚨 Connection error: {str(e)}"

print(verify_github_token())