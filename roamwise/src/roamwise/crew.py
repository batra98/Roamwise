from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
import panel as pn

chat_interface = pn.chat.ChatInterface()

from crewai.tasks.task_output import TaskOutput
from roamwise.tools.exa_search_tool import exa_search_tool
import os

# EXA-only configuration - simple and reliable
print("🚀 RoamWise initialized with EXA search tool")
print("   - Fast travel search using EXA neural search")
print("   - No browser automation - optimized for speed and reliability")

def print_output(output: TaskOutput):

    message = output.raw
    chat_interface.send(message, user=output.agent, respond=False)

@CrewBase
class Roamwise():
    """Roamwise crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools


    @agent
    def flight_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['flight_agent'], # type: ignore[index]
            tools=[exa_search_tool],  # Start with EXA only for initial search
            verbose=True
        )

    @agent
    def hotel_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['hotel_agent'], # type: ignore[index]
            tools=[exa_search_tool],  # Start with EXA only for initial search
            verbose=True
        )

    @agent
    def manager(self) -> Agent:
        return Agent(
            config=self.agents_config['manager'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def flight_search_task(self) -> Task:
        return Task(
            config=self.tasks_config['flight_search_task'], # type: ignore[index]
            callback=print_output,
        )

    @task
    def hotel_search_task(self) -> Task:
        return Task(
            config=self.tasks_config['hotel_search_task'], # type: ignore[index]
            callback=print_output,
        )

    @task
    def coordination_task(self) -> Task:
        return Task(
            config=self.tasks_config['coordination_task'], # type: ignore[index]
            output_file='travel_plan.md',
            callback=print_output,
            human_input=True,
            context=[self.flight_search_task(), self.hotel_search_task()],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Roamwise crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # All agents including manager
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential, # Sequential process to avoid hierarchical delegation issues
            verbose=True,
        )

    def run_initial_search(self, inputs: dict):
        """Run initial travel search using EXA only (fast)"""
        print("🚀 Running initial travel search (EXA only - fast)")
        print("   - Flight and hotel agents will use EXA search for broad coverage")
        return self.crew().kickoff(inputs=inputs)
