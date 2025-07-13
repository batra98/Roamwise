from dotenv import load_dotenv
import panel as pn
from datetime import datetime
import weave

load_dotenv()

# Initialize Weave for comprehensive logging (required)
weave.init("roamwise-travel-planner")

pn.extension(design="material")

from roamwise.crew import Roamwise
from roamwise.crew import chat_interface
import threading

from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
import time

def custom_ask_human_input(_self, final_answer: dict) -> str:
      
    global user_input

    chat_interface.send(final_answer, user="Assistant", respond=False)

    prompt = "Please provide feedback on the Final Result and the Agent's actions: "
    chat_interface.send(prompt, user="System", respond=False)

    while user_input == None:
        time.sleep(1)  

    human_comments = user_input
    user_input = None

    return human_comments


CrewAgentExecutorMixin._ask_human_input = custom_ask_human_input

user_input = None
crew_started = False
last_travel_results = None  # Store last travel search results for follow-up questions

def is_follow_up_question(message: str) -> bool:
    """Detect if message is a follow-up question about specific travel details"""
    follow_up_keywords = [
        "more details", "specific", "exact price", "availability", "book", "reserve",
        "amenities", "room types", "flight times", "seat selection", "baggage",
        "cancellation", "check-in", "check-out", "wifi", "breakfast", "parking"
    ]
    return any(keyword in message.lower() for keyword in follow_up_keywords)

@weave.op()
def initiate_chat(message):
    global crew_started, last_travel_results
    crew_started = True

    try:
        # All requests go through initial search - no separate detail extraction
        if last_travel_results and is_follow_up_question(message):
            print("🔍 Detected follow-up question - running new search with context")
            # Include previous results as context in the new search
            inputs = {
                "origin": "San Francisco",  # Will be extracted from message
                "destination": "Tokyo",  # Will be extracted from message
                "travel_dates": "2025-03-15 to 2025-03-20",  # Will be extracted from message
                "budget": "$2000",  # Will be extracted from message
                "preferences": f"Follow-up question: {message}. Previous results: {last_travel_results}",
                "current_year": datetime.now().year
            }
            with weave.attributes({
                "action": "follow_up_search_request",
                "user_query": message,
                "has_previous_results": bool(last_travel_results)
            }):
                roamwise = Roamwise()
                result = roamwise.run_initial_search(inputs=inputs)
                last_travel_results = str(result)
        else:
            print("🚀 Initial travel search request")
            # Parse travel request from message
            inputs = {
                "origin": "San Francisco",  # Will be extracted from message
                "destination": "Tokyo",  # Will be extracted from message
                "travel_dates": "2025-03-15 to 2025-03-20",  # Will be extracted from message
                "budget": "$2000",  # Will be extracted from message
                "preferences": message,  # Full user message as preferences
                "current_year": datetime.now().year
            }

            # Use Weave attributes to log metadata
            with weave.attributes({
                "action": "initial_travel_search",
                "inputs": inputs,
                "user_message": message
            }):
                roamwise = Roamwise()
                result = roamwise.run_initial_search(inputs=inputs)
                last_travel_results = str(result)  # Store for potential follow-up questions

        # Send results back to chat
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        # Log error using Weave attributes
        with weave.attributes({
            "action": "travel_planning_error",
            "error": str(e),
            "success": False
        }):
            chat_interface.send(error_msg, user="Assistant", respond=False)
    crew_started = False

def callback(contents: str, _user: str, _instance: pn.chat.ChatInterface):
    global crew_started, user_input, last_travel_results

    if not crew_started:
        thread = threading.Thread(target=initiate_chat, args=(contents,))
        thread.start()

    else:
        user_input = contents

chat_interface.callback = callback 

# Send welcome message
chat_interface.send(
    "Welcome to RoamWise! I'm your AI Travel Planning Assistant. Please provide your travel details: origin city, destination city, travel dates, budget, and any preferences. I'll coordinate my flight and hotel agents to find you the best options!",
    user="Assistant",
    respond=False
)

# Make it servable
chat_interface.servable()