import os
import json
import requests
import time
from dotenv import load_dotenv

def test_gemini_2_5_flash():
    """Test Gemini 2.5 Flash API connection"""
    load_dotenv('.env')
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ No API key found. Set GEMINI_API_KEY environment variable.")
        return False
    
    print("🔄 Testing Gemini 2.5 Flash API connection...")
    
    # API configuration
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    # Simple test payload
    payload = {
        "contents": [{
            "parts": [{
                "text": "Hello! Please respond with a simple JSON object containing just {'status': 'working'}."
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "topK": 1,
            "topP": 0.8,
            "maxOutputTokens": 100
        }
    }
    
    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        end_time = time.time()
        
        print(f"⏱️  Response time: {end_time - start_time:.2f} seconds")
        print(f"📊 Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if 'candidates' in result and len(result['candidates']) > 0:
                candidate = result['candidates'][0]
                
                if 'content' in candidate and 'parts' in candidate['content']:
                    response_text = candidate['content']['parts'][0]['text']
                    print(f"✅ Gemini 2.5 Flash is working!")
                    print(f"📝 Response: {response_text}")
                    
                    # Check token usage
                    if 'usageMetadata' in result:
                        usage = result['usageMetadata']
                        print(f"🔢 Token usage: {usage}")
                    
                    return True
                else:
                    print(f"❌ Unexpected response format: {result}")
                    return False
            else:
                print(f"❌ No candidates in response: {result}")
                return False
        else:
            print(f"❌ API error: {response.status_code}")
            print(f"📝 Error details: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 120 seconds")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    test_gemini_2_5_flash() 