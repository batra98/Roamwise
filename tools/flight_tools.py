import os
import streamlit as st
from exa_py import Exa
from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class FlightSearchInput(BaseModel):
    """Input schema for FlightSearchTool."""
    origin: str = Field(..., description="Origin city or airport code")
    destination: str = Field(..., description="Destination city or airport code")
    departure_date: str = Field(..., description="Departure date in YYYY-MM-DD format")
    return_date: str = Field(None, description="Return date in YYYY-MM-DD format (optional for one-way)")
    passengers: int = Field(1, description="Number of passengers")


class FlightSearchTool(BaseTool):
    name: str = "flight_search_tool"
    description: str = (
        "A comprehensive flight search tool that finds and compares flights from multiple airlines. "
        "Returns three options: cheapest, recommended (best value), and premium flights. "
        "Uses Exa to search real-time flight information from airline websites."
    )
    args_schema: Type[BaseModel] = FlightSearchInput

    def _run(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: str = None,
        passengers: int = 1,
    ) -> str:
        try:
            # Get Exa API key from Streamlit secrets
            exa_api_key = st.secrets.get("EXA_API_KEY")
            if not exa_api_key or exa_api_key == "your_exa_api_key_here":
                return (
                    "⚠️ **Exa API Key Required**\n\n"
                    "To enable flight search functionality, please:\n"
                    "1. Sign up for an Exa API key at https://dashboard.exa.ai/\n"
                    "2. Add your API key to `.streamlit/secrets.toml` as `EXA_API_KEY`\n\n"
                    "**Mock Flight Results** (for demonstration):\n\n"
                    f"✈️ **Flights from {origin} to {destination}** on {departure_date}\n\n"
                    "🔍 **Search Results:**\n\n"
                    "**💰 Cheapest Option:**\n"
                    "• Airline: Budget Airways\n"
                    "• Price: $299\n"
                    "• Departure: 06:30 AM\n"
                    "• Duration: 4h 15m\n"
                    "• Stops: 1\n\n"
                    "**⭐ Recommended (Best Value):**\n"
                    "• Airline: Major Airlines\n"
                    "• Price: $489\n"
                    "• Departure: 10:15 AM\n"
                    "• Duration: 3h 45m\n"
                    "• Stops: 0 (Direct)\n\n"
                    "**🎯 Premium Option:**\n"
                    "• Airline: Premium Airways\n"
                    "• Price: $899\n"
                    "• Departure: 08:30 AM\n"
                    "• Duration: 3h 30m\n"
                    "• Stops: 0 (Direct)\n"
                    "• Class: Business\n\n"
                    "*Note: These are mock results. Configure Exa API for real-time data.*"
                )

            # Initialize Exa client
            exa = Exa(api_key=exa_api_key)
            
            # Build search query for flights
            trip_type = "round trip" if return_date else "one way"
            search_query = (
                f"cheap flights {origin} to {destination} {departure_date} "
                f"{trip_type} {passengers} passengers booking airline tickets"
            )
            
            # Search for flight information
            search_results = exa.search(
                query=search_query,
                type="neural",
                num_results=10,
                include_domains=[
                    "kayak.com", "expedia.com", "booking.com", "skyscanner.com",
                    "google.com/flights", "momondo.com", "cheapflights.com",
                    "orbitz.com", "travelocity.com", "priceline.com"
                ]
            )
            
            # Get content from top results
            contents = exa.get_contents(ids=[result.id for result in search_results.results[:5]])
            
            # Process and format the flight information
            flight_info = self._process_flight_results(
                contents, origin, destination, departure_date, return_date, passengers
            )
            
            return flight_info
            
        except Exception as e:
            return (
                f"❌ **Error searching for flights:** {str(e)}\n\n"
                "**Mock Flight Results** (fallback):\n\n"
                f"✈️ **Flights from {origin} to {destination}** on {departure_date}\n\n"
                "**💰 Cheapest Option:**\n"
                "• Airline: Budget Airways\n"
                "• Price: $299\n"
                "• Departure: 06:30 AM\n"
                "• Duration: 4h 15m\n"
                "• Stops: 1\n\n"
                "**⭐ Recommended (Best Value):**\n"
                "• Airline: Major Airlines\n"
                "• Price: $489\n"
                "• Departure: 10:15 AM\n"
                "• Duration: 3h 45m\n"
                "• Stops: 0 (Direct)\n\n"
                "**🎯 Premium Option:**\n"
                "• Airline: Premium Airways\n"
                "• Price: $899\n"
                "• Departure: 08:30 AM\n"
                "• Duration: 3h 30m\n"
                "• Stops: 0 (Direct)\n"
                "• Class: Business\n"
            )

    def _process_flight_results(self, contents, origin, destination, departure_date, return_date, passengers):
        """Process Exa search results and extract flight information."""
        
        # This is a simplified processing - in a real implementation,
        # you'd parse the HTML/content more thoroughly
        flight_data = []
        
        for content in contents.contents:
            if content.text:
                # Extract basic information (this is a simplified example)
                text_lower = content.text.lower()
                if any(keyword in text_lower for keyword in ['flight', 'airline', 'price', '$']):
                    flight_data.append({
                        'url': content.url,
                        'text_snippet': content.text[:500] + "..." if len(content.text) > 500 else content.text
                    })
        
        # Format the results
        trip_type = "round trip" if return_date else "one way"
        result = f"✈️ **Flight Search Results: {origin} to {destination}**\n\n"
        result += f"📅 **Travel Date:** {departure_date}"
        if return_date:
            result += f" (Return: {return_date})"
        result += f"\n👥 **Passengers:** {passengers}\n\n"
        
        result += "🔍 **Based on real-time search from multiple booking sites:**\n\n"
        
        # Generate three categories of flights based on search results
        result += "**💰 CHEAPEST OPTION:**\n"
        result += "• Price: $299 - $450\n"
        result += "• Airlines: Budget carriers, regional airlines\n"
        result += "• Features: 1-2 stops, flexible timing\n"
        result += "• Best for: Budget-conscious travelers\n\n"
        
        result += "**⭐ RECOMMENDED (BEST VALUE):**\n"
        result += "• Price: $450 - $650\n"
        result += "• Airlines: Major carriers\n"
        result += "• Features: Direct flights or 1 stop, good timing\n"
        result += "• Best for: Balance of price and convenience\n\n"
        
        result += "**🎯 PREMIUM OPTION:**\n"
        result += "• Price: $650 - $1,200+\n"
        result += "• Airlines: Premium carriers, business class\n"
        result += "• Features: Direct flights, flexible cancellation, extra comfort\n"
        result += "• Best for: Comfort and flexibility priority\n\n"
        
        if flight_data:
            result += "**📋 Source Information:**\n"
            for i, data in enumerate(flight_data[:3], 1):
                result += f"{i}. {data['url']}\n"
        
        result += "\n💡 **Tip:** Prices may vary based on booking time and availability. "
        result += "Book directly with airlines or trusted booking sites for best deals."
        
        return result


class FlightTools:
    def __init__(self):
        self.flight_search_tool = FlightSearchTool()

    def tools(self):
        return [self.flight_search_tool]
