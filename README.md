# 🌍 RoamWise - Multi-Agent Travel Planner

**Intelligent flight search with AI-powered analysis and budget optimization**

RoamWise is a sophisticated travel planning tool that uses semantic search and AI agents to find the best flight deals, analyze your budget, and provide comprehensive travel insights.

## ✨ Features

- 🔍 **Intelligent Flight Search** - Semantic search using Exa API
- 💰 **Budget Analysis** - Smart budget breakdown and recommendations
- 📊 **Weave Integration** - Full observability and tracing
- 🎯 **Multi-Destination Search** - Compare flights to different cities
- 📱 **Beautiful CLI** - Rich, interactive command-line interface
- ⚡ **Fast & Reliable** - Optimized for performance

## 🚀 CLI Usage

### Interactive Search
```bash
# Start interactive flight search
poetry run roamwise search
```

### Command Line Search
```bash
# Search with all parameters
poetry run roamwise search \
  --from "San Francisco" \
  --to "Japan" \
  --departure "2025-12-09" \
  --days 5 \
  --budget 3000
```

### Examples

**Japan Trip (5 days, $3000 budget):**
```bash
poetry run roamwise search -f "San Francisco" -t "Japan" -d "2025-12-09" --days 5 -b 3000
```

**Europe Trip (7 days, $2500 budget):**
```bash
poetry run roamwise search -f "Los Angeles" -t "London" -d "2025-06-15" --days 7 -b 2500
```

## 📊 Sample Output

```
🌍 RoamWise - Multi-Agent Travel Planner

✈️ Trip Summary
🛫 San Francisco → 🛬 Japan
📅 Departure: 2025-12-09
💰 Budget: $3,000
📆 Duration: 5 days

🎉 Found 10 flights!

✈️ Flight Search Results
┏━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Airline ┃   Price ┃ Duration ┃ Source     ┃
┡━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ United  │  $283.0 │          │ Exa Search │
│ JAL     │     N/A │ 24h      │ Exa Search │
│ ANA     │     N/A │          │ Exa Search │
└─────────┴─────────┴──────────┴────────────┘

💰 Budget Analysis
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Category              ┃ Amount ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Total Budget          │ $3,000 │
│ Best Flight Price     │   $283 │
│ Remaining Budget      │ $2,717 │
│ Daily Budget (5 days) │   $543 │
└───────────────────────┴────────┘

✅ Excellent! Great flight prices with plenty left for hotels and activities
```

## 🏗️ Architecture

### Agents
- **Flight Agent**: Searches flights using Exa semantic search
- **Accommodation Agent**: Scrapes hotel data via Browserbase
- **Activity Agent**: Finds attractions and experiences
- **Itinerary Agent**: Optimizes daily schedules using geographic clustering

### Tools
- **Exa API**: Semantic web search for travel content
- **Browserbase**: Web scraping automation
- **Fly.io**: Compute functions for clustering algorithms
- **OpenAI**: Natural language processing and parsing
- **Weave**: Logging and tracing for all agent operations

## 📋 Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- Docker (for PostgreSQL database)
- API keys for: Exa, OpenAI, Browserbase, Fly.io, Weights & Biases

## 🛠️ Setup

### 1. Clone and Install Dependencies

```bash
git clone <repository-url>
cd roamwise
poetry install
```

### 2. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required API keys:
- `EXA_API_KEY`: Get from https://exa.ai
- `WANDB_API_KEY`: Get from https://wandb.ai
- `OPENAI_API_KEY`: Get from https://openai.com
- `BROWSERBASE_API_KEY`: Get from https://browserbase.com
- `FLY_API_TOKEN`: Get from https://fly.io

### 3. Start Database (Optional)

```bash
docker-compose up -d postgres
```

### 4. Test Flight Agent

```bash
poetry run python main.py
```

### 5. Interactive Flight Search

```bash
poetry run python main.py interactive
```

## 🧪 Testing the Flight Agent

The current implementation includes a working Flight Agent that you can test:

```python
from roamwise.agents.flight_agent import search_flights_with_agent

result = search_flights_with_agent(
    origin="San Francisco",
    destination="Tokyo",
    departure_date="2024-04-15",
    return_date="2024-04-22",
    budget=1200.0
)

print(result)
```

## 📊 Weave Logging

All agent operations are logged to Weights & Biases Weave for monitoring:

- Tool usage and performance
- Search queries and results
- Error tracking and debugging
- Agent decision making process

View your logs at: https://wandb.ai/your-username/roamwise

## 🔧 Current Status

✅ **Completed**:
- Project structure with Poetry
- Flight Agent with Exa integration
- Weave logging and tracing
- Configuration management
- Basic testing framework

🚧 **In Progress**:
- Additional agents (Hotel, Activity, Itinerary)
- Complete multi-agent workflow
- Database integration
- Advanced error handling

## 🚀 Next Steps

1. Test the Flight Agent with your API keys
2. Add Hotel Agent with Browserbase integration
3. Implement Activity and Itinerary agents
4. Create complete travel planning workflow
5. Add web interface for user interaction

## 🤝 Contributing

This project is part of the WeaveHacks hackathon. Feel free to contribute improvements and additional features!
