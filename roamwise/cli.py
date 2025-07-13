"""
RoamWise CLI - Multi-Agent Travel Planner with Enhanced Weave Logging
"""

import typer
import weave
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from datetime import datetime, timedelta
from typing import Optional
import sys

from .config import config
from .agents import CrewAITripOrchestrator
from .weave_functions import weave_trace, WeaveLogger
from .weave_monitoring import weave_monitor, log_user_interaction, monitor

# Initialize Rich console
console = Console()
app = typer.Typer(help="🌍 RoamWise - Multi-Agent Travel Planner")


def display_banner():
    """Display RoamWise banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🌍 RoamWise                               ║
    ║              Multi-Agent Travel Planner                     ║
    ║                                                              ║
    ║  ✈️  Intelligent Flight Search  🏨 Hotel Recommendations    ║
    ║  🎯 Budget Analysis            📊 Weave Integration         ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold blue")


def validate_date(date_str: str) -> bool:
    """Validate date format YYYY-MM-DD"""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def format_flight_results(flights: list) -> Table:
    """Format flight results in a nice table"""
    table = Table(title="✈️ Flight Search Results")
    
    table.add_column("Airline", style="cyan", no_wrap=True)
    table.add_column("Price", style="green", justify="right")
    table.add_column("Duration", style="yellow")
    table.add_column("Source", style="dim")
    
    for flight in flights:
        airline = flight.get('airline', 'Unknown')
        price = f"${flight.get('price', 'N/A')}" if flight.get('price') else "N/A"
        duration = flight.get('duration', 'N/A')
        source = "Flight Agent"
        
        table.add_row(airline, price, duration, source)
    
    return table


def display_budget_analysis(budget: float, min_price: float, avg_price: float, days: int):
    """Display budget analysis"""
    remaining = budget - min_price
    daily_budget = remaining / days if days > 0 else 0
    
    analysis_table = Table(title="💰 Budget Analysis")
    analysis_table.add_column("Category", style="bold")
    analysis_table.add_column("Amount", style="green", justify="right")
    
    analysis_table.add_row("Total Budget", f"${budget:,.0f}")
    analysis_table.add_row("Best Flight Price", f"${min_price:,.0f}")
    analysis_table.add_row("Average Flight Price", f"${avg_price:,.0f}")
    analysis_table.add_row("Remaining Budget", f"${remaining:,.0f}")
    analysis_table.add_row(f"Daily Budget ({days} days)", f"${daily_budget:,.0f}")
    
    console.print(analysis_table)
    
    # Budget recommendations
    if min_price <= budget * 0.3:
        console.print("✅ Excellent! Great flight prices with plenty left for hotels and activities", style="green")
    elif min_price <= budget * 0.5:
        console.print("👍 Good flight prices, comfortable budget remaining", style="yellow")
    elif min_price <= budget * 0.7:
        console.print("⚠️  Moderate flight prices, budget carefully for other expenses", style="orange")
    else:
        console.print("❌ High flight prices, consider adjusting dates or budget", style="red")


@app.command()
def search(
    origin: Optional[str] = typer.Option(None, "--from", "-f", help="Origin city"),
    destination: Optional[str] = typer.Option(None, "--to", "-t", help="Destination city"),
    departure: Optional[str] = typer.Option(None, "--departure", "-d", help="Departure date (YYYY-MM-DD)"),
    return_date: Optional[str] = typer.Option(None, "--return", "-r", help="Return date (YYYY-MM-DD)"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Maximum budget"),
    days: Optional[int] = typer.Option(None, "--days", help="Trip duration in days")
):
    """Search for flights with interactive prompts"""
    
    display_banner()
    
    try:
        # Initialize configuration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing RoamWise...", total=None)
            
            config.validate()
            config.init_weave()
            
            progress.update(task, description="✅ RoamWise initialized successfully!")
        
        console.print()
        
        # Interactive prompts if not provided
        if not origin:
            origin = Prompt.ask("🛫 [bold cyan]Origin city[/bold cyan]", default="San Francisco")
        
        if not destination:
            destination = Prompt.ask("🛬 [bold cyan]Destination city[/bold cyan]", default="Tokyo")
        
        if not departure:
            departure = Prompt.ask("📅 [bold cyan]Departure date[/bold cyan] (YYYY-MM-DD)", default="2025-12-09")
            while not validate_date(departure):
                console.print("❌ Invalid date format. Please use YYYY-MM-DD", style="red")
                departure = Prompt.ask("📅 [bold cyan]Departure date[/bold cyan] (YYYY-MM-DD)")
        
        if not return_date and not days:
            trip_type = Prompt.ask(
                "🔄 [bold cyan]Trip type[/bold cyan]", 
                choices=["round-trip", "one-way"], 
                default="round-trip"
            )
            
            if trip_type == "round-trip":
                if not days:
                    days = int(Prompt.ask("📆 [bold cyan]Trip duration[/bold cyan] (days)", default="5"))
                
                # Calculate return date
                dep_date = datetime.strptime(departure, "%Y-%m-%d")
                ret_date = dep_date + timedelta(days=days)
                return_date = ret_date.strftime("%Y-%m-%d")

        # Ensure days is set for one-way trips
        if not days:
            days = 7  # Default to 7 days for one-way trips

        if not budget:
            budget = float(Prompt.ask("💰 [bold cyan]Maximum budget[/bold cyan] ($)", default="3000"))
        
        # Display trip summary
        console.print()
        trip_panel = Panel(
            f"🛫 [bold]{origin}[/bold] → 🛬 [bold]{destination}[/bold]\n"
            f"📅 Departure: [green]{departure}[/green]\n" +
            (f"📅 Return: [green]{return_date}[/green]\n" if return_date else "") +
            f"💰 Budget: [green]${budget:,.0f}[/green]" +
            (f"\n📆 Duration: [yellow]{days} days[/yellow]" if days else ""),
            title="✈️ Trip Summary",
            border_style="blue"
        )
        console.print(trip_panel)
        console.print()
        
        # Confirm search
        if not Confirm.ask("🔍 [bold cyan]Search for flights?[/bold cyan]", default=True):
            console.print("👋 Search cancelled. Safe travels!", style="yellow")
            return
        
        # Initialize CrewAI Trip Orchestrator
        orchestrator = CrewAITripOrchestrator()

        # Search flights using Flight Agent with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🤖 CrewAI Agents working...", total=None)

            trip_result = orchestrator.plan_trip(
                origin=origin,
                destination=destination,
                departure_date=departure,
                return_date=return_date,
                budget=budget,
                days=days  # Pass the days parameter
            )

            progress.update(task, description="✅ CrewAI Agents completed!")
        
        console.print()
        
        # Display CrewAI results
        if trip_result.get('success'):
            console.print(f"🎉 CrewAI Multi-Agent System completed successfully!")
            console.print()

            # Show CrewAI result
            crew_result = trip_result.get('crew_result', '')
            console.print(f"🤖 CrewAI Result:")

            # Extract text content from CrewAI result object
            if hasattr(crew_result, 'raw'):
                result_text = str(crew_result.raw)
            elif hasattr(crew_result, '__str__'):
                result_text = str(crew_result)
            else:
                result_text = "Result completed successfully"

            console.print(Panel(result_text, title="✈️ Flight Search Results", border_style="green"))
            console.print()

            console.print(f"✅ Flight search and analysis completed by specialized agents")
            console.print(f"📊 All agent interactions logged in Weave for full traceability")

        else:
            console.print("❌ CrewAI execution failed", style="red")
            if trip_result.get('error'):
                console.print(f"Error: {trip_result['error']}", style="dim red")
            
            # Weave trace info
            console.print()
            console.print(
                Panel(
                    f"📊 All searches logged in Weave for analysis\n"
                    f"🔗 View traces: [link]https://wandb.ai/gbatra3-uw-madison/Roamwise/weave[/link]",
                    title="📈 Analytics",
                    border_style="dim"
                )
            )

    
    except KeyboardInterrupt:
        console.print("\n👋 Search cancelled. Safe travels!", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        sys.exit(1)


@app.command()
def agents():
    """Show available agents and their capabilities"""
    display_banner()

    try:
        config.validate()
        config.init_weave()

        orchestrator = CrewAITripOrchestrator()
        available_agents = orchestrator.get_available_agents()

        console.print("🤖 Available CrewAI Agents", style="bold blue")
        console.print()

        for agent_name, agent_info in available_agents.items():
            agent_panel = Panel(
                f"🎯 [bold]Role:[/bold] {agent_info['role']}\n"
                f"📋 [bold]Goal:[/bold] {agent_info['goal']}\n"
                f"🔧 [bold]Tools:[/bold] {', '.join(agent_info['tools'])}",
                title=f"🤖 {agent_name.replace('_', ' ').title()}",
                border_style="cyan"
            )
            console.print(agent_panel)
            console.print()

        # Show CrewAI capabilities
        capabilities_table = Table(title="🚀 CrewAI Multi-Agent Capabilities")
        capabilities_table.add_column("Agent", style="cyan")
        capabilities_table.add_column("Capability", style="white")
        capabilities_table.add_column("Tool Used", style="green")

        capabilities_table.add_row("Flight Search Agent", "Semantic flight search", "Exa API")
        capabilities_table.add_row("Orchestrator Agent", "Result analysis", "Flight Analysis Tool")
        capabilities_table.add_row("Orchestrator Agent", "Budget recommendations", "Python Logic")
        capabilities_table.add_row("Orchestrator Agent", "Trip coordination", "Agent Delegation")

        console.print(capabilities_table)

    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")


@app.command()
def version():
    """Show RoamWise version"""
    console.print("🌍 RoamWise v1.0.0 - Multi-Agent Travel Planner", style="bold blue")


@app.command()
@weave_monitor("cli_health_check")
def health():
    """Check system health and performance metrics"""
    console.print(Panel.fit("🏥 RoamWise Health Check", style="bold green"))

    try:
        # Initialize Weave
        weave.init(project_name="gbatra3-uw-madison/Roamwise")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("🔍 Checking system health...", total=None)

            # Get health status
            health_status = monitor.get_health_status()

            progress.update(task, description="✅ Health check completed!")

        # Display health status
        status = health_status["status"]
        status_color = {
            "healthy": "green",
            "warning": "yellow",
            "critical": "red"
        }.get(status, "white")

        console.print(f"\n🏥 System Status: [{status_color}]{status.upper()}[/{status_color}]")

        # Display alerts if any
        alerts = health_status.get("alerts", [])
        if alerts:
            console.print("\n⚠️ [bold yellow]Alerts:[/bold yellow]")
            for alert in alerts:
                console.print(f"  • {alert}")
        else:
            console.print("\n✅ [green]No alerts - system running smoothly[/green]")

        # Display performance summary
        summary = health_status["summary"]

        perf_table = Table(title="📊 Performance Summary")
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green", justify="right")

        perf_table.add_row("Total Operations", str(summary["total_operations"]))
        perf_table.add_row("Total Errors", str(summary["total_errors"]))
        perf_table.add_row("Error Rate", f"{summary['error_rate']:.2%}")
        perf_table.add_row("System Uptime", f"{summary['system_uptime_seconds']:.1f}s")

        recent = summary["recent_activity"]
        perf_table.add_row("Recent Operations (5min)", str(recent["operations_last_5min"]))
        perf_table.add_row("Recent Errors (5min)", str(recent["errors_last_5min"]))
        perf_table.add_row("Avg Duration (5min)", f"{recent['avg_duration_last_5min']:.2f}s")

        console.print(perf_table)

        # Display operation counts
        if summary["operation_counts"]:
            op_table = Table(title="🔧 Operation Counts")
            op_table.add_column("Operation", style="cyan")
            op_table.add_column("Count", style="green", justify="right")

            for op, count in summary["operation_counts"].items():
                op_table.add_row(op, str(count))

            console.print(op_table)

    except Exception as e:
        console.print(f"\n[red]❌ Health check failed: {str(e)}[/red]")


@app.command()
@weave_monitor("cli_api_health")
def api_health():
    """Check external API health and connectivity"""
    console.print(Panel.fit("🌐 API Health Check", style="bold cyan"))

    try:
        # Initialize Weave
        weave.init(project_name="gbatra3-uw-madison/Roamwise")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("🔍 Checking API connectivity...", total=None)

            # Import here to avoid circular imports
            from .weave_functions import check_api_health

            health_results = check_api_health()

            progress.update(task, description="✅ API health check completed!")

        # Display API health results
        apis = health_results.get("apis", {})

        api_table = Table(title="🌐 API Health Status")
        api_table.add_column("API", style="cyan")
        api_table.add_column("Status", style="bold")
        api_table.add_column("Response Time", style="yellow", justify="right")
        api_table.add_column("Details", style="dim")

        for api_name, api_status in apis.items():
            status = api_status.get("status", "unknown")
            status_color = {
                "healthy": "green",
                "unhealthy": "yellow",
                "error": "red"
            }.get(status, "white")

            response_time = api_status.get("response_time_seconds", "N/A")
            response_time_str = f"{response_time:.3f}s" if isinstance(response_time, (int, float)) else str(response_time)

            details = ""
            if status == "error":
                details = api_status.get("error", "Unknown error")[:50]
            elif "status_code" in api_status:
                details = f"HTTP {api_status['status_code']}"

            api_table.add_row(
                api_name.upper(),
                f"[{status_color}]{status.upper()}[/{status_color}]",
                response_time_str,
                details
            )

        console.print(api_table)

        # Overall health summary
        healthy_apis = sum(1 for api in apis.values() if api.get("status") == "healthy")
        total_apis = len(apis)

        if healthy_apis == total_apis:
            console.print(f"\n✅ [green]All {total_apis} APIs are healthy[/green]")
        else:
            console.print(f"\n⚠️ [yellow]{healthy_apis}/{total_apis} APIs are healthy[/yellow]")

    except Exception as e:
        console.print(f"\n[red]❌ API health check failed: {str(e)}[/red]")


if __name__ == "__main__":
    app()
