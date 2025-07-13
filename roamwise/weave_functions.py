"""
Enhanced Weave logging and tracing for RoamWise
Comprehensive observability with detailed metrics, error tracking, and performance monitoring
"""

import os
import re
import weave
import httpx
import traceback
import functools
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone
import time
import json

from .config import config


# Enhanced Weave Logging Utilities
class WeaveLogger:
    """Enhanced Weave logging utility with comprehensive metrics and error tracking"""

    @staticmethod
    def log_operation_start(operation_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Log the start of an operation with parameters"""
        return {
            "operation": operation_name,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "parameters": params,
            "status": "started"
        }

    @staticmethod
    def log_operation_success(start_log: Dict[str, Any], result: Any, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log successful operation completion with results and metrics"""
        end_time = datetime.now(timezone.utc)
        start_time = datetime.fromisoformat(start_log["start_time"])
        duration = (end_time - start_time).total_seconds()

        return {
            **start_log,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "status": "success",
            "result_summary": WeaveLogger._summarize_result(result),
            "metrics": metrics or {},
            "success": True
        }

    @staticmethod
    def log_operation_error(start_log: Dict[str, Any], error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Log operation failure with detailed error information"""
        end_time = datetime.now(timezone.utc)
        start_time = datetime.fromisoformat(start_log["start_time"])
        duration = (end_time - start_time).total_seconds()

        return {
            **start_log,
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_traceback": traceback.format_exc(),
            "context": context or {},
            "success": False
        }

    @staticmethod
    def _summarize_result(result: Any) -> Dict[str, Any]:
        """Create a summary of the result for logging"""
        if isinstance(result, dict):
            return {
                "type": "dict",
                "keys": list(result.keys()),
                "size": len(result)
            }
        elif isinstance(result, list):
            return {
                "type": "list",
                "length": len(result),
                "sample": result[:3] if result else []
            }
        elif isinstance(result, str):
            return {
                "type": "string",
                "length": len(result),
                "preview": result[:100] + "..." if len(result) > 100 else result
            }
        else:
            return {
                "type": str(type(result).__name__),
                "value": str(result)[:100]
            }


def weave_trace(operation_name: str = None, log_params: bool = True, log_result: bool = True):
    """
    Enhanced Weave decorator for comprehensive operation tracing

    Args:
        operation_name: Custom name for the operation (defaults to function name)
        log_params: Whether to log function parameters
        log_result: Whether to log function results
    """
    def decorator(func):
        @functools.wraps(func)
        @weave.op()
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            # Prepare parameters for logging
            params = {}
            if log_params:
                params = {
                    "args": [str(arg)[:100] for arg in args],  # Truncate long args
                    "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}
                }

            # Log operation start
            start_log = WeaveLogger.log_operation_start(op_name, params)

            try:
                # Execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Prepare metrics
                metrics = {
                    "execution_time_seconds": execution_time,
                    "function_name": func.__name__,
                    "module": func.__module__
                }

                # Log success
                success_log = WeaveLogger.log_operation_success(start_log, result if log_result else "Result logged separately", metrics)

                # Return result with logging metadata
                if isinstance(result, dict):
                    result["_weave_trace"] = success_log

                return result

            except Exception as e:
                # Log error
                error_log = WeaveLogger.log_operation_error(start_log, e, {
                    "function_name": func.__name__,
                    "module": func.__module__
                })

                # Re-raise the exception
                raise

        return wrapper
    return decorator


@weave_trace("flight_search_exa", log_params=True, log_result=True)
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
    
    # Enhanced logging with detailed metrics
    api_start_time = time.time()

    try:
        # Call Exa API directly with enhanced logging
        with httpx.Client() as client:
            request_payload = {
                "query": query,
                "type": "neural",
                "useAutoprompt": True,
                "numResults": config.MAX_RESULTS_PER_SEARCH,
                "contents": {
                    "text": True,
                    "highlights": True
                }
            }

            # Log API request details
            api_request_log = {
                "api": "exa",
                "endpoint": "search",
                "payload_size": len(json.dumps(request_payload)),
                "query_length": len(query),
                "max_results": config.MAX_RESULTS_PER_SEARCH,
                "timeout": config.EXA_TIMEOUT
            }

            response = client.post(
                "https://api.exa.ai/search",
                headers={
                    "Authorization": f"Bearer {config.EXA_API_KEY[:8]}...",  # Masked for security
                    "Content-Type": "application/json"
                },
                json=request_payload,
                timeout=config.EXA_TIMEOUT
            )

            api_response_time = time.time() - api_start_time
            response.raise_for_status()

            result = response.json()
            results = result.get("results", [])

            # Enhanced flight extraction with metrics
            extraction_start_time = time.time()
            flights = []
            extraction_errors = []

            for i, res in enumerate(results):
                try:
                    flight_info = _extract_flight_info_simple(res)
                    if flight_info:
                        flights.append(flight_info)
                except Exception as e:
                    extraction_errors.append({
                        "result_index": i,
                        "error": str(e),
                        "result_title": res.get("title", "Unknown")[:50]
                    })

            extraction_time = time.time() - extraction_start_time

            # Comprehensive result with enhanced metrics
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
                "api_metrics": {
                    **api_request_log,
                    "response_time_seconds": api_response_time,
                    "response_status": response.status_code,
                    "response_size_bytes": len(response.content)
                },
                "extraction_metrics": {
                    "extraction_time_seconds": extraction_time,
                    "successful_extractions": len(flights),
                    "failed_extractions": len(extraction_errors),
                    "extraction_errors": extraction_errors,
                    "success_rate": len(flights) / len(results) if results else 0
                },
                "performance_metrics": {
                    "total_time_seconds": time.time() - api_start_time,
                    "results_per_second": len(results) / api_response_time if api_response_time > 0 else 0,
                    "flights_per_second": len(flights) / extraction_time if extraction_time > 0 else 0
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

    except Exception as e:
        api_error_time = time.time() - api_start_time

        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "flights_found": 0,
            "query_used": query,
            "search_params": {
                "origin": str(origin),
                "destination": str(destination),
                "departure_date": str(departure_date),
                "return_date": str(return_date) if return_date else None,
                "budget": float(budget) if budget else None
            },
            "error_metrics": {
                "error_time_seconds": api_error_time,
                "error_traceback": traceback.format_exc()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@weave_trace("flight_info_extraction", log_params=False, log_result=True)
def _extract_flight_info_simple(result: dict) -> Optional[dict]:
    """Extract flight information from Exa search result with enhanced logging"""

    extraction_start = time.time()
    extraction_metrics = {
        "patterns_tried": 0,
        "patterns_matched": 0,
        "text_length": 0,
        "extraction_success": False
    }

    try:
        title = result.get('title', '')
        text = result.get('text', '')
        url = result.get('url', '')
        highlights = result.get('highlights', [])

        # Combine all text for analysis
        full_text = f"{title} {text} {' '.join(highlights)}"
        extraction_metrics["text_length"] = len(full_text)

        # Enhanced price extraction with logging
        price_patterns = [
            r'\$(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?) USD',
            r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?) dollars?'
        ]

        price = None
        price_matches = []
        for i, pattern in enumerate(price_patterns):
            extraction_metrics["patterns_tried"] += 1
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                extraction_metrics["patterns_matched"] += 1
                price_matches.extend(matches)
                try:
                    price_str = matches[0].replace(',', '')
                    price = float(price_str)
                    break
                except ValueError:
                    continue

        # Enhanced airline extraction
        airlines = [
            "United", "Delta", "American", "Southwest", "JetBlue",
            "Alaska", "Spirit", "Frontier", "Hawaiian", "Allegiant",
            "Emirates", "Lufthansa", "British Airways", "Air France",
            "KLM", "Singapore Airlines", "Cathay Pacific", "ANA",
            "JAL", "Qatar Airways", "Turkish Airlines", "LEVEL",
            "TAP", "Iberia", "Norwegian", "Finnair"
        ]

        airline = None
        airlines_found = []
        text_lower = full_text.lower()
        for a in airlines:
            if a.lower() in text_lower:
                airlines_found.append(a)
                if not airline:  # Take the first match
                    airline = a

        # Enhanced duration extraction
        duration_patterns = [
            r'(\d+h\s*\d*m?)',
            r'(\d+\s*hours?\s*\d*\s*minutes?)',
            r'(\d+hr\s*\d*min)'
        ]

        duration = None
        duration_matches = []
        for pattern in duration_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            if matches:
                duration_matches.extend(matches)
                if not duration:
                    duration = matches[0]

        extraction_time = time.time() - extraction_start

        # Create enhanced flight info dict with extraction metadata
        flight = {
            "airline": airline,
            "price": price,
            "duration": duration,
            "booking_url": url,
            "source": "exa_search",
            "title": title[:100] if title else None,
            "extraction_metadata": {
                "extraction_time_seconds": extraction_time,
                "price_matches_found": price_matches,
                "airlines_found": airlines_found,
                "duration_matches_found": duration_matches,
                "text_analysis": {
                    "title_length": len(title),
                    "text_length": len(text),
                    "highlights_count": len(highlights),
                    "total_text_length": len(full_text)
                },
                "extraction_metrics": extraction_metrics
            }
        }

        # Determine if extraction was successful
        has_useful_info = bool(airline or price or duration)
        extraction_metrics["extraction_success"] = has_useful_info

        if has_useful_info:
            return flight

        return None

    except Exception as e:
        extraction_time = time.time() - extraction_start
        extraction_metrics.update({
            "extraction_time_seconds": extraction_time,
            "error": str(e),
            "error_type": type(e).__name__
        })
        return None


@weave_trace("flight_results_analysis", log_params=True, log_result=True)
def analyze_flight_results(flights: list) -> dict:
    """Analyze flight search results with comprehensive insights and metrics"""

    analysis_start = time.time()

    if not flights:
        return {
            "total_flights": 0,
            "analysis": "No flights found",
            "analysis_time_seconds": time.time() - analysis_start,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    # Enhanced data extraction with validation
    prices = []
    airlines = []
    durations = []
    booking_urls = []
    extraction_metadata = []

    for i, flight in enumerate(flights):
        # Extract and validate price
        price = flight.get("price")
        if price is not None and isinstance(price, (int, float)) and price > 0:
            prices.append(price)

        # Extract airline
        airline = flight.get("airline")
        if airline and isinstance(airline, str):
            airlines.append(airline)

        # Extract duration
        duration = flight.get("duration")
        if duration and isinstance(duration, str):
            durations.append(duration)

        # Extract booking URL
        url = flight.get("booking_url")
        if url and isinstance(url, str):
            booking_urls.append(url)

        # Collect extraction metadata if available
        metadata = flight.get("extraction_metadata", {})
        if metadata:
            extraction_metadata.append({
                "flight_index": i,
                **metadata
            })

    # Advanced price analysis
    price_analysis = {}
    if prices:
        sorted_prices = sorted(prices)
        price_analysis = {
            "min": min(prices),
            "max": max(prices),
            "avg": sum(prices) / len(prices),
            "median": sorted_prices[len(sorted_prices) // 2],
            "std_dev": (sum((p - sum(prices) / len(prices)) ** 2 for p in prices) / len(prices)) ** 0.5,
            "price_distribution": {
                "under_500": len([p for p in prices if p < 500]),
                "500_to_1000": len([p for p in prices if 500 <= p < 1000]),
                "1000_to_1500": len([p for p in prices if 1000 <= p < 1500]),
                "over_1500": len([p for p in prices if p >= 1500])
            }
        }

    # Airline analysis
    airline_analysis = {}
    if airlines:
        airline_counts = {}
        for airline in airlines:
            airline_counts[airline] = airline_counts.get(airline, 0) + 1

        airline_analysis = {
            "unique_airlines": list(set(airlines)),
            "airline_frequency": airline_counts,
            "most_common_airline": max(airline_counts.items(), key=lambda x: x[1])[0],
            "airline_diversity": len(set(airlines)) / len(airlines)
        }

    # Data quality metrics
    data_quality = {
        "price_coverage": len(prices) / len(flights) * 100 if flights else 0,
        "airline_coverage": len(airlines) / len(flights) * 100 if flights else 0,
        "duration_coverage": len(durations) / len(flights) * 100 if flights else 0,
        "url_coverage": len(booking_urls) / len(flights) * 100 if flights else 0,
        "overall_quality_score": (
            len(prices) + len(airlines) + len(durations) + len(booking_urls)
        ) / (len(flights) * 4) * 100 if flights else 0
    }

    # Performance metrics from extraction metadata
    performance_metrics = {}
    if extraction_metadata:
        extraction_times = [m.get("extraction_time_seconds", 0) for m in extraction_metadata]
        text_lengths = [m.get("text_analysis", {}).get("total_text_length", 0) for m in extraction_metadata]

        performance_metrics = {
            "avg_extraction_time": sum(extraction_times) / len(extraction_times) if extraction_times else 0,
            "total_extraction_time": sum(extraction_times),
            "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
            "extraction_efficiency": len(flights) / sum(extraction_times) if sum(extraction_times) > 0 else 0
        }

    analysis_time = time.time() - analysis_start

    comprehensive_analysis = {
        "total_flights": len(flights),
        "flights_with_prices": len(prices),
        "flights_with_airlines": len(airlines),
        "flights_with_durations": len(durations),
        "flights_with_urls": len(booking_urls),
        "price_analysis": price_analysis,
        "airline_analysis": airline_analysis,
        "data_quality": data_quality,
        "performance_metrics": performance_metrics,
        "analysis_metadata": {
            "analysis_time_seconds": analysis_time,
            "extraction_metadata_available": len(extraction_metadata),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

    return comprehensive_analysis


# Additional Weave-enhanced utility functions
@weave_trace("api_health_check", log_params=True, log_result=True)
def check_api_health() -> dict:
    """Check the health and connectivity of external APIs"""

    health_results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "apis": {}
    }

    # Check EXA API
    try:
        start_time = time.time()
        with httpx.Client() as client:
            response = client.get("https://api.exa.ai/", timeout=10)
            exa_time = time.time() - start_time

        health_results["apis"]["exa"] = {
            "status": "healthy" if response.status_code < 500 else "unhealthy",
            "response_time_seconds": exa_time,
            "status_code": response.status_code
        }
    except Exception as e:
        health_results["apis"]["exa"] = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

    # Check Weave connectivity
    try:
        # Simple check by attempting to log a test operation
        test_log = WeaveLogger.log_operation_start("health_check", {"test": True})
        health_results["apis"]["weave"] = {
            "status": "healthy",
            "test_log_created": bool(test_log)
        }
    except Exception as e:
        health_results["apis"]["weave"] = {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

    return health_results
