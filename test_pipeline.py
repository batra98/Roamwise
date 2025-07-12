#!/usr/bin/env python3
"""
Test Pipeline for Trip Planner with Gemini
This script tests the complete CrewAI pipeline to ensure all components work correctly
with the Gemini integration.
"""

import sys
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from crewai import Crew, LLM
from trip_agents import TripAgents
from trip_tasks import TripTasks

def test_gemini_connection():
    """Test basic Gemini connection"""
    print("🔍 Testing Gemini Connection...")
    try:
        import google.generativeai as genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ GEMINI_API_KEY not found in environment variables")
            return False
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say 'Hello, I'm ready to help plan trips!' in exactly 10 words.")
        
        print(f"✅ Gemini Connection: SUCCESS")
        print(f"📝 Response: {response.text[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Gemini Connection: FAILED - {str(e)}")
        return False

def test_crewai_llm():
    """Test CrewAI LLM integration with Gemini"""
    print("\n🤖 Testing CrewAI LLM Integration...")
    try:
        llm = LLM(model="gemini/gemini-2.0-flash")
        print("✅ CrewAI LLM: Successfully initialized")
        return llm
    except Exception as e:
        print(f"❌ CrewAI LLM: FAILED - {str(e)}")
        return None

def test_agents_creation(llm):
    """Test agent creation with Gemini"""
    print("\n👥 Testing Agent Creation...")
    try:
        agents = TripAgents(llm=llm)
        
        # Test each agent creation
        city_selector = agents.city_selection_agent()
        print("✅ City Selection Agent: Created successfully")
        
        local_expert = agents.local_expert()
        print("✅ Local Expert Agent: Created successfully")
        
        travel_concierge = agents.travel_concierge()
        print("✅ Travel Concierge Agent: Created successfully")
        
        return agents
    except Exception as e:
        print(f"❌ Agent Creation: FAILED - {str(e)}")
        return None

def test_tasks_creation(agents):
    """Test task creation"""
    print("\n📋 Testing Task Creation...")
    try:
        tasks = TripTasks()
        
        # Sample trip parameters
        origin = "San Francisco, CA"
        cities = "Paris, France"
        interests = "art, food, museums, walking tours"
        date_range = "2025-09-01 to 2025-09-08"
        
        # Test task creation
        identify_task = tasks.identify_task(
            agents.city_selection_agent(),
            origin, cities, interests, date_range
        )
        print("✅ Identify Task: Created successfully")
        
        gather_task = tasks.gather_task(
            agents.local_expert(),
            origin, interests, date_range
        )
        print("✅ Gather Task: Created successfully")
        
        plan_task = tasks.plan_task(
            agents.travel_concierge(),
            origin, interests, date_range
        )
        print("✅ Plan Task: Created successfully")
        
        return [identify_task, gather_task, plan_task], (origin, cities, interests, date_range)
    except Exception as e:
        print(f"❌ Task Creation: FAILED - {str(e)}")
        return None, None

def test_simple_crew_execution(agents, tasks_info):
    """Test a simple crew execution with limited scope"""
    print("\n🚀 Testing Simple Crew Execution...")
    try:
        tasks_list, trip_params = tasks_info
        origin, cities, interests, date_range = trip_params
        
        # Create a simplified crew with just the first task
        simple_crew = Crew(
            agents=[agents.city_selection_agent()],
            tasks=[tasks_list[0]],  # Only the identify task
            verbose=True
        )
        
        print(f"🎯 Testing with: {origin} → {cities}")
        print(f"📅 Dates: {date_range}")
        print(f"🎨 Interests: {interests}")
        print("\n⏳ Running simplified crew (this may take 1-2 minutes)...")
        
        result = simple_crew.kickoff()
        
        print("\n✅ Crew Execution: SUCCESS")
        print("📄 Result Preview:")
        result_str = str(result)
        print(f"{result_str[:300]}...")
        return True
        
    except Exception as e:
        print(f"❌ Crew Execution: FAILED - {str(e)}")
        return False

def run_full_pipeline_test():
    """Run the complete pipeline test"""
    print("🏖️  Trip Planner Pipeline Test - Gemini Integration")
    print("=" * 60)
    
    # Check environment variables
    required_keys = ['GEMINI_API_KEY', 'SERPER_API_KEY', 'BROWSERLESS_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
        print("Please check your .env file")
        return False
    
    print("✅ Environment variables: All present")
    
    # Test sequence
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Gemini connection
    if test_gemini_connection():
        tests_passed += 1
    
    # Test 2: CrewAI LLM
    llm = test_crewai_llm()
    if llm:
        tests_passed += 1
    else:
        print("⏹️  Stopping tests due to LLM failure")
        return False
    
    # Test 3: Agent creation
    agents = test_agents_creation(llm)
    if agents:
        tests_passed += 1
    else:
        print("⏹️  Stopping tests due to agent creation failure")
        return False
    
    # Test 4: Task creation
    tasks_info = test_tasks_creation(agents)
    if tasks_info[0]:
        tests_passed += 1
    else:
        print("⏹️  Stopping tests due to task creation failure")
        return False
    
    # Test 5: Simple crew execution
    if test_simple_crew_execution(agents, tasks_info):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The Gemini integration is working correctly.")
        print("✅ Ready to run the full Streamlit application!")
        return True
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_full_pipeline_test()
    sys.exit(0 if success else 1)
