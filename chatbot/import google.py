import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

def test_gemini_api(api_key: str):
    """Test if Gemini API key is valid and working"""
    try:
        print("Testing Gemini API connection...")
        
        # Test with direct google.generativeai
        print("\n1. Testing with google.generativeai...")
        genai.configure(api_key=api_key)
        
        # List available models
        models = genai.list_models()
        print(f"Available models: {[m.name for m in models]}")
        
        # Test a simple generation
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello, who are you?")
        print(f"\nDirect API response: {response.text}")
        
        # Test with LangChain wrapper
        print("\n2. Testing with LangChain wrapper...")
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7
        )
        
        langchain_response = llm.invoke("Tell me a fun fact about AI")
        print(f"LangChain response: {langchain_response.content}")
        
        return True, "API key is valid and working!"
    
    except Exception as e:
        return False, f"API test failed: {str(e)}"

if __name__ == "__main__":
    # Get API key - try environment variable first
    import os
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        api_key = input("Enter your Gemini API key: ").strip()
    
    print("\nStarting Gemini API test...")
    success, message = test_gemini_api(api_key)
    
    print("\nTest Results:")
    print(f"Success: {success}")
    print(f"Message: {message}")