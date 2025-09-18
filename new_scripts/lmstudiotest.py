import requests
import json
import os

def test_lm_studio_connection():
    """
    Test connection to LM Studio API and send prompt from debug_ai_prompt.txt
    """
    # API endpoint
    url = "http://127.0.0.1:1234/v1/chat/completions"
    
    # Read the prompt from debug_ai_prompt.txt
    prompt_file = r"e:\MIMIC Research\Code\Report-Generation-and-Disease-Classification\new_scripts\debug_ai_prompt.txt"
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        print(f"✅ Successfully loaded prompt from {prompt_file}")
        print(f"📄 Prompt length: {len(prompt_content)} characters")
    except FileNotFoundError:
        print(f"❌ Error: Could not find prompt file at {prompt_file}")
        return False
    except Exception as e:
        print(f"❌ Error reading prompt file: {e}")
        return False
    
    # Prepare the request payload
    payload = {
        "model": "medgemma-27b-it",
        "messages": [
            {
                "role": "user",
                "content": prompt_content
            }
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("\n🔄 Testing connection to LM Studio...")
    print(f"🎯 Target URL: {url}")
    # print(f"🤖 Model: llama-4-scout-17b-16e-instruct")
    
    try:
        # Send the request
        print("\n📤 Sending request...")
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        # Check response status
        if response.status_code == 200:
            print("✅ Connection successful!")
            
            # Parse the response
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                ai_response = response_data['choices'][0]['message']['content']
                print("\n📋 AI Response:")
                print("-" * 50)
                print(ai_response)
                print("-" * 50)
                
                # Save response to file for analysis
                output_file = r"e:\MIMIC Research\Code\Report-Generation-and-Disease-Classification\new_scripts\lm_studio_response.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"LM Studio Response - {requests.utils.default_user_agent()}\n")
                    f.write(f"Model: llama-4-scout-17b-16e-instruct\n")
                    f.write(f"Timestamp: {requests.utils.default_user_agent()}\n\n")
                    f.write(ai_response)
                
                print(f"\n💾 Response saved to: {output_file}")
                return True
            else:
                print("❌ No response content found in API response")
                print(f"Raw response: {response_data}")
                return False
                
        else:
            print(f"❌ Connection failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to LM Studio")
        print("💡 Make sure LM Studio is running on http://127.0.0.1:1234")
        print("💡 Check if the model 'llama-4-scout-17b-16e-instruct' is loaded")
        return False
    except requests.exceptions.Timeout:
        print("❌ Timeout Error: Request took too long")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Error: {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON Decode Error: {e}")
        print(f"Raw response: {response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_lm_studio_health():
    """
    Check if LM Studio is running and accessible
    """
    health_url = "http://127.0.0.1:1234/v1/models"
    
    try:
        print("\n🏥 Checking LM Studio health...")
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            models_data = response.json()
            print("✅ LM Studio is running!")
            
            if 'data' in models_data:
                print("\n📋 Available models:")
                for model in models_data['data']:
                    model_id = model.get('id', 'Unknown')
                    print(f"  • {model_id}")
                    
                # Check if our target model is available
                target_model = "llama-4-scout-17b-16e-instruct"
                available_models = [model.get('id', '') for model in models_data['data']]
                
                if target_model in available_models:
                    print(f"\n✅ Target model '{target_model}' is available!")
                    return True
                else:
                    print(f"\n⚠️  Target model '{target_model}' not found in available models")
                    print("💡 Make sure the model is loaded in LM Studio")
                    return False
            else:
                print("⚠️  No model data found in response")
                return False
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 LM Studio Connection Test")
    print("=" * 40)
    
    # First check if LM Studio is running
    if check_lm_studio_health():
        print("\n" + "=" * 40)
        # If health check passes, test the actual prompt
        success = test_lm_studio_connection()
        
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n❌ Test failed!")
    else:
        print("\n❌ Cannot proceed - LM Studio health check failed")
        print("\n💡 Troubleshooting steps:")
        print("   1. Make sure LM Studio is running")
        print("   2. Check if the server is listening on http://127.0.0.1:1234")
        print("   3. Load the 'llama-4-scout-17b-16e-instruct' model")
        print("   4. Ensure the model is ready for inference")