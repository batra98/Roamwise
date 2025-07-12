"""
Configuration management for RoamWise
Simplified configuration focusing on core tools: Weave, Exa, Browserbase, Fly.io, Google A2A
"""

import os
import ssl
import urllib3
from dotenv import load_dotenv
import weave

# Load environment variables
load_dotenv()

# Fix SSL certificate issues for corporate networks
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set environment variables to disable SSL verification
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''


class Config:
    """Application configuration"""
    
    # Core Tool API Keys
    EXA_API_KEY = os.getenv("EXA_API_KEY")
    BROWSERBASE_API_KEY = os.getenv("BROWSERBASE_API_KEY")
    BROWSERBASE_PROJECT_ID = os.getenv("BROWSERBASE_PROJECT_ID")
    FLY_API_TOKEN = os.getenv("FLY_API_TOKEN")
    FLY_APP_NAME = os.getenv("FLY_APP_NAME")

    # Google A2A Configuration
    GOOGLE_A2A_CLIENT_ID = os.getenv("GOOGLE_A2A_CLIENT_ID")
    GOOGLE_A2A_CLIENT_SECRET = os.getenv("GOOGLE_A2A_CLIENT_SECRET")

    # Google Gemini API Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAL_SKGgflwXejY_-tgMGJgHI9SrRLQ7vU")
    
    # Weave Configuration
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    WANDB_PROJECT = os.getenv("WANDB_PROJECT", "gbatra3-uw-madison/Roamwise")
    
    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # API Timeouts (seconds)
    EXA_TIMEOUT = 30
    BROWSERBASE_TIMEOUT = 60
    FLY_TIMEOUT = 45
    
    # Safety Limits
    MAX_TOOL_CALLS = 10
    MAX_RESULTS_PER_SEARCH = 10
    
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        # Check EXA_API_KEY
        if not cls.EXA_API_KEY:
            raise ValueError("Missing required environment variable: EXA_API_KEY")

        # Check WANDB_API_KEY (can be in environment or ~/.netrc)
        if not cls.WANDB_API_KEY:
            # Check if it might be in ~/.netrc
            try:
                import netrc
                netrc_data = netrc.netrc()
                if 'api.wandb.ai' in netrc_data.hosts:
                    print("✅ WANDB credentials found in ~/.netrc")
                else:
                    raise ValueError("WANDB_API_KEY not found in environment or ~/.netrc")
            except (FileNotFoundError, netrc.NetrcParseError):
                raise ValueError("WANDB_API_KEY not found in environment or ~/.netrc")
    
    @classmethod
    def init_weave(cls):
        """Initialize Weave logging with proper error handling"""
        # Weave can use credentials from ~/.netrc even if WANDB_API_KEY env var is not set
        try:
            # Initialize Weave with project name
            weave.init(cls.WANDB_PROJECT)
            print("✅ Weave initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Weave initialization failed: {e}")
            # For SSL issues, provide helpful guidance
            if "SSL" in str(e) or "certificate" in str(e).lower():
                print("💡 SSL Certificate issue detected.")
                print("   This might be due to corporate network restrictions.")
                print("   Try running from a different network or contact your IT team.")
            raise


# Initialize configuration
config = Config()
