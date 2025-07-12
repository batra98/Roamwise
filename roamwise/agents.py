"""
CrewAI Multi-Agent System for RoamWise
Using only allowed tools: Weave, Exa, Browserbase, Fly.io, Google A2A
"""

import weave
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import EXASearchTool

from .weave_functions import search_flights_weave, analyze_flight_results
from .config import config


# CrewAI LLM Configuration
def create_gemini_llm():
    """Create Gemini LLM for CrewAI agents"""
    return LLM(
        model="gemini/gemini-2.0-flash",  # Fast, cost-effective Gemini model
        temperature=0.7,  # Balanced creativity and consistency
        api_key=config.GEMINI_API_KEY
    )


# CrewAI Tools Setup
def create_exa_search_tool():
    """Create EXA search tool for flight searches"""
    return EXASearchTool(api_key=config.EXA_API_KEY)


# CrewAI Agents
def create_flight_search_agent():
    """Create CrewAI flight search agent with Gemini LLM"""
    return Agent(
        role="Flight Search Specialist",
        goal="Find flight information using semantic search for travel planning",
        backstory="""You are an expert flight search specialist who uses Exa semantic search
        to find flight-related information from the web. You search for flight prices,
        airlines, and booking options.""",
        tools=[create_exa_search_tool()],
        llm=create_gemini_llm(),  # Use Gemini LLM
        verbose=True,
        allow_delegation=False
    )


def create_orchestrator_agent():
    """Create CrewAI orchestrator agent with Gemini LLM - EXTRACT SPECIFIC FLIGHT OPTIONS"""
    return Agent(
        role="Flight Analysis Specialist",
        goal="Extract specific flight options with prices, airlines, and booking links from search results",
        backstory="""You are an expert flight analyst who specializes in parsing flight search results
        and extracting the most important information for travelers. You identify the best deals,
        compare options, and present clear recommendations with specific prices and booking links.
        You excel at finding the key details travelers need to make booking decisions.""",
        tools=[],  # NO TOOLS - only analysis
        llm=create_gemini_llm(),  # Use Gemini LLM
        verbose=True,
        allow_delegation=False  # NO DELEGATION - prevents loops
    )


# CrewAI Tasks
def create_flight_search_task(user_request: Dict[str, Any]):
    """Create flight search task for CrewAI"""
    origin = user_request.get('origin')
    destination = user_request.get('destination')
    departure_date = user_request.get('departure_date')
    return_date = user_request.get('return_date')
    budget = user_request.get('budget')

    search_query = f"flights from {origin} to {destination} departing {departure_date}"
    if return_date:
        search_query += f" returning {return_date}"
    if budget:
        search_query += f" under ${budget}"
    search_query += " booking prices airlines deals"

    return Task(
        description=f"""
        Search for flight information using Exa semantic search.

        Search Query: "{search_query}"

        Find comprehensive flight information including:
        - Flight prices and booking options
        - Different airlines serving this route
        - Flight deals and promotions
        - Booking websites and travel agencies

        Focus on finding current, accurate flight information for the specified route and dates.
        """,
        expected_output="Detailed flight search results with prices, airlines, and booking information",
        agent=create_flight_search_agent()
    )


def create_analysis_task(user_request):
    """Create analysis task for CrewAI - Extract specific flight options"""
    return Task(
        description=f"""
        Analyze the flight search results from the Flight Search Specialist and extract SPECIFIC flight options.

        USER REQUEST DETAILS:
        - Route: {user_request.get('from_city', 'Unknown')} → {user_request.get('to_city', 'Unknown')}
        - Budget: ${user_request.get('budget', 'Not specified')}
        - Duration: {user_request.get('days', 'Unknown')} days
        - Departure: {user_request.get('departure_date', 'Unknown')}

        From the search results provided, identify and present:

        **TOP FLIGHT RECOMMENDATIONS:**
        1. **Best Price Option**: Extract the lowest price found, which airline, and booking link
        2. **Best Value Option**: Consider price vs. convenience (direct flights, good airlines)
        3. **Premium Option**: Higher-end airlines or better schedules if available

        **FORMAT YOUR RESPONSE EXACTLY LIKE THIS:**

        🎯 **FLIGHT RECOMMENDATIONS FOR {user_request.get('from_city', 'ORIGIN').upper()} → {user_request.get('to_city', 'DESTINATION').upper()}**

        ✈️ **Option 1: Best Price**
        - **Price**: $XXX (or €XXX)
        - **Airline**: [Airline Name]
        - **Book at**: [Website/Link]
        - **Why**: [Brief reason]

        ✈️ **Option 2: Best Value**
        - **Price**: $XXX
        - **Airline**: [Airline Name]
        - **Book at**: [Website/Link]
        - **Why**: [Brief reason]

        💰 **BUDGET ANALYSIS**
        - Budget: ${user_request.get('budget', 'Not specified')}
        - Route: {user_request.get('from_city', 'Unknown')} → {user_request.get('to_city', 'Unknown')}
        - Duration: {user_request.get('days', 'Unknown')} days
        - Calculate percentage of budget used by best price option
        - Calculate remaining budget per day for hotels/activities

        🎯 **RECOMMENDATION**: [Your top pick and why]

        Extract this information directly from the search results provided. If specific prices aren't visible, mention "Visit [website] for current pricing" but still recommend the best options found.
        """,
        expected_output="Formatted flight recommendations with specific prices, airlines, and booking links",
        agent=create_orchestrator_agent(),
        context=[]  # Will be populated with flight search results
    )


# CrewAI Orchestrator
class CrewAITripOrchestrator:
    """CrewAI-based trip orchestrator"""

    def __init__(self):
        self.flight_search_agent = create_flight_search_agent()
        self.orchestrator_agent = create_orchestrator_agent()

    @weave.op()
    def plan_trip(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Plan trip using CrewAI multi-agent system
        """

        user_request = {
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "return_date": return_date,
            "budget": budget
        }

        try:
            # Create tasks
            flight_task = create_flight_search_task(user_request)
            analysis_task = create_analysis_task(user_request)

            # Set task context (analysis depends on flight search)
            analysis_task.context = [flight_task]

            # Create crew with constraints to prevent loops
            crew = Crew(
                agents=[self.flight_search_agent, self.orchestrator_agent],
                tasks=[flight_task, analysis_task],
                process=Process.sequential,
                verbose=True,
                max_iter=3,  # Maximum 3 iterations to prevent loops
                memory=False  # Disable memory to prevent confusion
            )

            # Execute crew
            result = crew.kickoff()

            return {
                "success": True,
                "crew_result": result,
                "orchestrator": "CrewAI",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "orchestrator": "CrewAI",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    def get_available_agents(self) -> Dict[str, Any]:
        """Get information about available CrewAI agents"""
        return {
            "flight_search_agent": {
                "role": self.flight_search_agent.role,
                "goal": self.flight_search_agent.goal,
                "tools": [tool.name for tool in self.flight_search_agent.tools]
            },
            "orchestrator_agent": {
                "role": self.orchestrator_agent.role,
                "goal": self.orchestrator_agent.goal,
                "tools": [tool.name for tool in self.orchestrator_agent.tools]
            }
        }
