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

    # Google Gemini API Configuration (CrewAI recommended)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    # Google Vertex AI Configuration (backward compatibility)
    GOOGLE_VERTEX_API_KEY = os.getenv("GOOGLE_VERTEX_API_KEY")
    
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
        """Initialize Weave logging with enhanced error handling and monitoring"""
        import time
        from datetime import datetime, timezone

        # Weave can use credentials from ~/.netrc even if WANDB_API_KEY env var is not set
        try:
            # Initialize with comprehensive settings and timing
            init_start = time.time()
            weave.init(cls.WANDB_PROJECT)
            init_time = time.time() - init_start

            # Create initialization log
            init_log = {
                "status": "success",
                "project": cls.WANDB_PROJECT,
                "init_time_seconds": init_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "weave_version": getattr(weave, "__version__", "unknown"),
                "config_validation": {
                    "exa_api_key_present": bool(cls.EXA_API_KEY),
                    "wandb_api_key_present": bool(cls.WANDB_API_KEY),
                    "gemini_api_key_present": bool(cls.GEMINI_API_KEY),
                    "vertex_api_key_present": bool(cls.GOOGLE_VERTEX_API_KEY),
                    "browserbase_api_key_present": bool(cls.BROWSERBASE_API_KEY),
                    "fly_api_token_present": bool(cls.FLY_API_TOKEN)
                },
                "environment_info": {
                    "debug_mode": cls.DEBUG,
                    "log_level": cls.LOG_LEVEL,
                    "ssl_verification_disabled": True,
                    "max_tool_calls": cls.MAX_TOOL_CALLS,
                    "max_results_per_search": cls.MAX_RESULTS_PER_SEARCH
                }
            }

            # Use Weave to log its own initialization (meta-logging)
            @weave.op()
            def log_weave_initialization(init_data):
                """Log Weave initialization details"""
                return init_data

            log_weave_initialization(init_log)

            print(f"✅ Weave initialized successfully in {init_time:.3f}s")
            print(f"📊 Project: {cls.WANDB_PROJECT}")
            print(f"🔧 Configuration validated: {sum(init_log['config_validation'].values())}/5 API keys present")

            return True

        except Exception as e:
            init_time = time.time() - init_start if 'init_start' in locals() else 0

            # Enhanced error logging
            error_log = {
                "status": "failed",
                "project": cls.WANDB_PROJECT,
                "init_time_seconds": init_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "ssl_related": "SSL" in str(e) or "certificate" in str(e).lower(),
                "config_state": {
                    "wandb_api_key_env": bool(cls.WANDB_API_KEY),
                    "debug_mode": cls.DEBUG
                }
            }

            print(f"❌ Weave initialization failed after {init_time:.3f}s: {e}")

            # Enhanced SSL guidance
            if error_log["ssl_related"]:
                print("💡 SSL Certificate issue detected.")
                print("   This might be due to corporate network restrictions.")
                print("   Current SSL settings: verification disabled")
                print("   Try running from a different network or contact your IT team.")
                print("   Alternative: Check if ~/.netrc contains valid WANDB credentials")

            # Check for common issues
            if "authentication" in str(e).lower() or "unauthorized" in str(e).lower():
                print("🔑 Authentication issue detected.")
                print("   Please check your WANDB_API_KEY or ~/.netrc file")
                print("   Get your API key from: https://wandb.ai/authorize")

            raise


# Initialize configuration
config = Config()
