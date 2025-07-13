import weave
import os
import time
import traceback
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from crewai import Agent, Task, Crew, Process, LLM

from .config import config
from .weave_functions import weave_trace, WeaveLogger

# CrewAI LLM Configuration
def create_gemini_llm():
    """Create Gemini LLM using CrewAI recommended configuration"""
    try:
        # Check for GEMINI_API_KEY first (CrewAI recommended)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            # Fallback to GOOGLE_VERTEX_API_KEY for backward compatibility
            gemini_api_key = os.getenv("GOOGLE_VERTEX_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_VERTEX_API_KEY environment variable not set")

        # Use CrewAI recommended format: gemini/model-name
        return LLM(
            model="gemini/gemini-1.5-flash",  # CrewAI format with provider prefix
            temperature=0.2,  # Low temperature for consistent output
            max_tokens=1500,  # Conservative token limit
            timeout=45  # Reasonable timeout
        )
    except Exception as e:
        print(f"❌ Gemini LLM creation failed: {e}")
        print("🔄 Trying fallback configuration...")

        # Fallback: Try with basic configuration
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_VERTEX_API_KEY", "")
        try:
            return LLM(
                model="gemini/gemini-1.5-flash",
                temperature=0.3
            )
        except Exception as fallback_error:
            print(f"❌ Fallback LLM creation also failed: {fallback_error}")
            print("💡 Please check your GEMINI_API_KEY and ensure it's valid for Gemini API")
            print("💡 Get your API key from: https://aistudio.google.com/apikey")
            # Return a basic configuration as last resort
            return LLM(
                model="gemini/gemini-1.5-flash"
            )