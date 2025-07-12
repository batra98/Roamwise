"""
Weave-compatible functions for RoamWise
These functions use simple Python types for perfect Weave integration
"""

import os
import re
import weave
import httpx
from typing import Optional
from datetime import datetime, timezone

from .config import config


@weave.op()
def search_flights_weave(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: Optional[str] = None,
    budget: Optional[float] = None
) -> dict:
    """
    Search for flights using Exa API with full Weave logging
    
    Args:
        origin: Origin city or airport code
        destination: Destination city or airport code
        departure_date: Departure date (YYYY-MM-DD)
        return_date: Return date (YYYY-MM-DD), optional
        budget: Maximum budget, optional
        
    Returns:
        Dictionary with flight search results (simple types only)
    """
    
    # Build search query
    query_parts = [
        f"flights from {origin} to {destination}",
        f"departing {departure_date}"
    ]
    
    if return_date:
        query_parts.append(f"returning {return_date}")
    
    if budget:
        query_parts.append(f"under ${budget}")
    
    query_parts.extend([
        "booking prices",
        "airlines",
        "flight deals",
        "travel booking"
    ])
    
    query = " ".join(query_parts)
    
    try:
        # Call Exa API directly
        with httpx.Client() as client:
            response = client.post(
                "https://api.exa.ai/search",
                headers={
                    "Authorization": f"Bearer {config.EXA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "query": query,
                    "type": "neural",
                    "useAutoprompt": True,
                    "numResults": config.MAX_RESULTS_PER_SEARCH,
                    "contents": {
                        "text": True,
                        "highlights": True
                    }
                },
                timeout=config.EXA_TIMEOUT
            )
            response.raise_for_status()
            
            result = response.json()
            results = result.get("results", [])
            
            # Extract flight info using simple logic
            flights = []
            for res in results:
                flight_info = _extract_flight_info_simple(res)
                if flight_info:
                    flights.append(flight_info)
            
            return {
                "success": True,
                "flights_found": len(flights),
                "query_used": query,
                "search_params": {
                    "origin": str(origin),
                    "destination": str(destination),
                    "departure_date": str(departure_date),
                    "return_date": str(return_date) if return_date else None,
                    "budget": float(budget) if budget else None
                },
                "flights": flights,
                "total_exa_results": len(results),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "flights_found": 0,
            "query_used": query,
            "search_params": {
                "origin": str(origin),
                "destination": str(destination),
                "departure_date": str(departure_date),
                "return_date": str(return_date) if return_date else None,
                "budget": float(budget) if budget else None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def _extract_flight_info_simple(result: dict) -> Optional[dict]:
    """Extract flight information from Exa search result"""
    
    try:
        title = result.get('title', '')
        text = result.get('text', '')
        url = result.get('url', '')
        highlights = result.get('highlights', [])
        
        # Combine all text for analysis
        full_text = f"{title} {text} {' '.join(highlights)}"
        
        # Extract price using regex
        price_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?) USD',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?) dollars?'
        ]
        
        price = None
        for pattern in price_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                try:
                    price_str = matches[0].replace(',', '')
                    price = float(price_str)
                    break
                except ValueError:
                    continue
        
        # Extract airline
        airlines = [
            "United", "Delta", "American", "Southwest", "JetBlue",
            "Alaska", "Spirit", "Frontier", "Hawaiian", "Allegiant",
            "Emirates", "Lufthansa", "British Airways", "Air France",
            "KLM", "Singapore Airlines", "Cathay Pacific", "ANA",
            "JAL", "Qatar Airways", "Turkish Airlines"
        ]
        
        airline = None
        text_lower = full_text.lower()
        for a in airlines:
            if a.lower() in text_lower:
                airline = a
                break
        
        # Extract duration
        duration_patterns = [
            r'(\d+h\s*\d*m?)',
            r'(\d+\s*hours?\s*\d*\s*minutes?)',
            r'(\d+hr\s*\d*min)'
        ]
        
        duration = None
        for pattern in duration_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                duration = matches[0]
                break
        
        # Create flight info dict
        flight = {
            "airline": airline,
            "price": price,
            "duration": duration,
            "booking_url": url,
            "source": "exa_search",
            "title": title[:100] if title else None  # Truncate title
        }
        
        # Only return flights with some useful information
        if airline or price or duration:
            return flight
        
        return None
        
    except Exception:
        return None


@weave.op()
def analyze_flight_results(flights: list) -> dict:
    """Analyze flight search results and provide insights"""
    
    if not flights:
        return {
            "total_flights": 0,
            "analysis": "No flights found"
        }
    
    # Extract prices
    prices = [f.get("price") for f in flights if f.get("price")]
    airlines = [f.get("airline") for f in flights if f.get("airline")]
    durations = [f.get("duration") for f in flights if f.get("duration")]
    
    analysis = {
        "total_flights": len(flights),
        "flights_with_prices": len(prices),
        "flights_with_airlines": len(airlines),
        "flights_with_durations": len(durations),
        "unique_airlines": list(set(airlines)) if airlines else [],
        "price_range": {
            "min": min(prices) if prices else None,
            "max": max(prices) if prices else None,
            "avg": sum(prices) / len(prices) if prices else None
        },
        "data_quality": {
            "price_coverage": len(prices) / len(flights) * 100 if flights else 0,
            "airline_coverage": len(airlines) / len(flights) * 100 if flights else 0,
            "duration_coverage": len(durations) / len(flights) * 100 if flights else 0
        }
    }
    
    return analysis
