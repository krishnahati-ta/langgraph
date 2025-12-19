# 🦜🕸️LangGraph

⚡ Build language agents as graphs ⚡

## Overview

LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to other LLM frameworks, it offers these core benefits: cycles, controllability, and persistence. LangGraph allows you to define flows that involve cycles, essential for most agentic architectures, differentiating it from DAG-based solutions. As a very low-level framework, it provides fine-grained control over both the flow and state of your application, crucial for creating reliable agents. Additionally, LangGraph includes built-in persistence, enabling advanced human-in-the-loop and memory features.

LangGraph is inspired by [Pregel](https://research.google/pubs/pub37252/) and [Apache Beam](https://beam.apache.org/). The public interface draws inspiration from [NetworkX](https://networkx.org/documentation/latest/). LangGraph is built by LangChain Inc, the creators of LangChain, but can be used without LangChain.

To learn more about LangGraph, check out our first LangChain Academy course, [Introduction to LangGraph](https://academy.langchain.com/courses/intro-to-langgraph), available for free.

---
*Note: This branch (code_agent) is being used for development and testing purposes.*

## Key Features

- **Cycles and Branching**: Implement complex agent behaviors with cycles and conditional branching.
- **Persistence**: Built-in persistence for long-running, multi-conversation applications.
- **Human-in-the-Loop**: Interrupt and approve steps for human oversight.
- **Streaming Support**: Stream outputs as they are produced by each node.
- **Integration with LangChain**: Seamlessly integrate with LangChain and LangSmith.

## Installation

```bash
pip install langgraph
```

## Example

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": [("assistant", "Hello! How can I help you today?")]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()
```

## Documentation

For detailed documentation, tutorials, and examples, visit:
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.