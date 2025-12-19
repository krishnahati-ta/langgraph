# 🦜🕸️LangGraph

⚡ Build language agents as graphs ⚡

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures, differentiating it from DAG-based solutions. As a very low-level framework, it provides fine-grained control over both the flow and state of your application, crucial for creating reliable agents. Additionally, LangGraph includes built-in persistence, enabling advanced human-in-the-loop and memory features.

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.

To learn more about LangGraph, check out our first LangChain Academy course, *Introduction to LangGraph*, available for free [here](https://academy.langchain.com/courses/intro-to-langgraph).

## Key Features

- **Cycles and Branching**: Implement loops and conditionals in your apps.
- **Persistence**: Automatically save state after each step in the graph. Pause and resume the graph execution at any point to support error recovery, human-in-the-loop workflows, time travel, and more.
- **Human-in-the-Loop**: Interrupt graph execution to approve or edit next action planned by the agent.
- **Streaming Support**: Stream outputs as they are produced by each node (including token streaming).
- **Integration with LangChain**: LangGraph integrates seamlessly with [LangChain](https://github.com/langchain-ai/langchain/) and [LangSmith](https://smith.langchain.com/) (but does not require them).

## Installation

```bash
pip install langgraph
```

## Example

One of the central concepts of LangGraph is state. Each graph execution creates a state that is passed between nodes in the graph as they execute, and each node updates this state with its return value after it executes. The way that the graph updates its state is defined by either the type of graph chosen or a custom function.

Let's take a look at a simple example of an agent that can use a search tool.

```python
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode


# Define the tools for the agent to use
tools = [TavilySearchResults(max_results=2)]
tool_node = ToolNode(tools)

model = ChatOpenAI(temperature=0, streaming=True)

# Define the function that determines whether to continue or not
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edge(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

# Use the Runnable
final_state = app.invoke(
    {"messages": [HumanMessage(content="what is the weather in sf")]},
    config={"configurable": {"thread_id": 42}}
)
final_state["messages"][-1].content
```

Now when we run the graph:

```python
from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
for output in app.stream(inputs, config={"configurable": {"thread_id": 42}}):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")
```

This will output:
```
Output from node 'agent':
---
{'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_H7fABDuzEau48T10Qn0Lsh0D', 'function': {'arguments': '{"query":"current weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]})]}

---

Output from node 'tools':
---
{'messages': [ToolMessage(content='[{"url": "https://www.weatherapi.com/", "content": "{\\'location\\': {\\'name\\': \\'San Francisco\\', \\'region\\': \\'California\\', \\'country\\': \\'United States of America\\', \\'lat\\': 37.78, \\'lon\\': -122.42, \\'tz_id\\': \\'America/Los_Angeles\\', \\'localtime_epoch\\': 1707638400, \\'localtime\\': \\'2024-02-11 04:40\\'}, \\'current\\': {\\'last_updated_epoch\\': 1707637800, \\'last_updated\\': \\'2024-02-11 04:30\\', \\'temp_c\\': 11.1, \\'temp_f\\': 52.0, \\'is_day\\': 0, \\'condition\\': {\\'text\\': \\'Partly cloudy\\', \\'icon\\': \\'//cdn.weatherapi.com/weather/64x64/night/116.png\\', \\'code\\': 1003}, \\'wind_mph\\': 5.6, \\'wind_kph\\': 9.0, \\'wind_degree\\': 56, \\'wind_dir\\': \\'ENE\\', \\'pressure_mb\\': 1024.0, \\'pressure_in\\': 30.24, \\'precip_mm\\': 0.0, \\'precip_in\\': 0.0, \\'humidity\\': 88, \\'cloud\\': 25, \\'feelslike_c\\': 9.5, \\'feelslike_f\\': 49.2, \\'vis_km\\': 16.0, \\'vis_miles\\': 9.0, \\'uv\\': 1.0, \\'gust_mph\\': 12.5, \\'gust_kph\\': 20.1}}"}]', name='tavily_search_results_json', tool_call_id='call_H7fABDuzEau48T10Qn0Lsh0D')]}

---

Output from node 'agent':
---
{'messages': [AIMessage(content='The current weather in San Francisco is partly cloudy with a temperature of 52°F (11.1°C). The humidity is at 88%, and there are light winds coming from the ENE at 5.6 mph. It feels like 49.2°F (9.5°C) outside.\n\nWould you like more details about the weather or forecast for San Francisco?')]}

---
```

## Documentation

* [Tutorials](https://langchain-ai.github.io/langgraph/tutorials/): Learn to build with LangGraph through guided examples.
* [How-to Guides](https://langchain-ai.github.io/langgraph/how-tos/): Accomplish specific things within LangGraph, from streaming, to adding memory & persistence, to common design patterns (branching, subgraphs, etc.), these guides show how to use LangGraph to create powerful applications.
* [Conceptual Guides](https://langchain-ai.github.io/langgraph/concepts/): In-depth explanations of the key concepts and principles behind LangGraph, such as nodes, edges, state and more.
* [API Reference](https://langchain-ai.github.io/langgraph/reference/): Review important classes and methods, simple examples of how to use the graph and checkpointing APIs, higher-level prebuilt components and more.
* [LangGraph Academy](https://academy.langchain.com/courses/intro-to-langgraph): A free course to get you started building with LangGraph.

## Contributing

For more information on how to contribute, see [here](https://github.com/langchain-ai/langgraph/blob/main/CONTRIBUTING.md).

*Note: This branch (code_agent) is being used for development and testing purposes.*