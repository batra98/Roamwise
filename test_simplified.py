#!/usr/bin/env python3
"""
Simplified Test Pipeline for Gemini Integration
Tests individual components without CrewAI to verify Gemini integration works
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_setup():
    """Test that all required environment variables are present"""
    print("🔍 Testing Environment Setup...")
    
    required_keys = ['GEMINI_API_KEY', 'SERPER_API_KEY', 'BROWSERLESS_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        value = os.getenv(key)
        if not value:
            missing_keys.append(key)
        else:
            print(f"✅ {key}: Present")
    
    if missing_keys:
        print(f"❌ Missing keys: {', '.join(missing_keys)}")
        return False
    
    print("✅ All environment variables present")
    return True

def test_gemini_basic():
    """Test basic Gemini API connection"""
    print("\n🤖 Testing Gemini API Connection...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say hello and confirm you're ready to help with travel planning!")
        
        print("✅ Gemini API: Connected successfully")
        print(f"📝 Response preview: {response.text[:150]}...")
        return True
        
    except Exception as e:
        print(f"❌ Gemini API: Failed - {str(e)}")
        return False

def test_search_tools():
    """Test the search tools functionality"""
    print("\n🔍 Testing Search Tools...")
    
    try:
        # Import the search tools
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from tools.search_tools import SearchTools
        
        search_tool = SearchTools()
        result = search_tool._run("Paris France travel attractions")
        
        if result and len(result) > 50:
            print("✅ Search Tools: Working correctly")
            print(f"📝 Sample result: {result[:100]}...")
            return True
        else:
            print("❌ Search Tools: No results or error")
            return False
            
    except Exception as e:
        print(f"❌ Search Tools: Failed - {str(e)}")
        return False

def test_calculator_tools():
    """Test the calculator tools functionality"""
    print("\n🧮 Testing Calculator Tools...")
    
    try:
        from tools.calculator_tools import CalculatorTools
        
        calc_tool = CalculatorTools()
        result = calc_tool._run("200 * 7 + 100")
        
        if result == 1500:
            print("✅ Calculator Tools: Working correctly")
            print(f"📝 Test calculation (200*7+100): {result}")
            return True
        else:
            print(f"❌ Calculator Tools: Unexpected result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Calculator Tools: Failed - {str(e)}")
        return False

def test_langchain_google_genai():
    """Test LangChain Google GenAI integration"""
    print("\n🔗 Testing LangChain Google GenAI...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        response = llm.invoke("What are the top 3 attractions in Paris? Be brief.")
        
        print("✅ LangChain Google GenAI: Working correctly")
        print(f"📝 Response preview: {response.content[:150]}...")
        return True
        
    except Exception as e:
        print(f"❌ LangChain Google GenAI: Failed - {str(e)}")
        return False

def test_simple_trip_planning():
    """Test a simple trip planning scenario using Gemini directly"""
    print("\n🏖️ Testing Simple Trip Planning with Gemini...")
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = """
        You are a travel expert. Plan a 3-day trip to Paris, France for someone from San Francisco 
        who loves art and food. Include:
        1. Top 3 attractions to visit
        2. 2 restaurant recommendations
        3. Best transportation method
        
        Keep it concise (under 200 words).
        """
        
        response = model.generate_content(prompt)
        
        if response.text and len(response.text) > 100:
            print("✅ Simple Trip Planning: Working correctly")
            print(f"📝 Trip plan preview:\n{response.text[:300]}...")
            return True
        else:
            print("❌ Simple Trip Planning: No response or too short")
            return False
            
    except Exception as e:
        print(f"❌ Simple Trip Planning: Failed - {str(e)}")
        return False

def run_simplified_tests():
    """Run all simplified tests"""
    print("🏖️  Trip Planner Simplified Test Suite - Gemini Integration")
    print("=" * 65)
    
    tests = [
        test_environment_setup,
        test_gemini_basic,
        test_search_tools,
        test_calculator_tools,
        test_langchain_google_genai,
        test_simple_trip_planning
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()  # Add spacing between tests
    
    # Summary
    print("=" * 65)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Gemini integration is working correctly.")
        print("✅ Core components are ready for the Streamlit application!")
        return True
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = run_simplified_tests()
    
    if success:
        print("\n🚀 Ready to run the Streamlit app!")
        print("💡 To start the app, run:")
        print("   streamlit run streamlit_app.py")
    
    sys.exit(0 if success else 1)
