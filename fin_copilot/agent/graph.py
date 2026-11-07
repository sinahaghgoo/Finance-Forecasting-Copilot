from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from fin_copilot.agent.prompts import SYSTEM_PROMPT
from fin_copilot.agent.tools import ALL_TOOLS
from fin_copilot.config import load_config


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_llm(provider: str | None = None, model_name: str | None = None, temperature: float = 0.0) -> BaseChatModel:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    cfg = load_config().agent
    provider = os.getenv("FINCOPILOT_MODEL_PROVIDER", provider or cfg.model_provider)
    model_name = os.getenv("FINCOPILOT_MODEL_NAME", model_name or cfg.model_name)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(model=model_name, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        return ChatAnthropic(model=model_name, temperature=temperature)
    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise RuntimeError(
                "langchain-ollama is required for the Ollama provider. "
                "Install with: pip install langchain-ollama"
            ) from e
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=host, temperature=temperature)
    raise ValueError(f"Unknown provider: {provider}")


@dataclass
class CompiledAgent:
    graph: Any
    tool_names: list[str]

    def invoke(self, question: str) -> dict:
        initial = {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]}
        final_state = self.graph.invoke(initial)
        return _trace(final_state, question)


def build_agent(llm: BaseChatModel | None = None, tools: list | None = None) -> CompiledAgent:
    tools = tools or ALL_TOOLS
    llm = llm if llm is not None else _build_llm()
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    def agent_node(state: AgentState) -> dict:
        reply = llm_with_tools.invoke(state["messages"])
        return {"messages": [reply]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    compiled = graph.compile()
    return CompiledAgent(graph=compiled, tool_names=[t.name for t in tools])


def _trace(state: dict, question: str) -> dict:
    messages = state["messages"]
    tool_calls, tool_results = [], []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                tool_calls.append({"id": tc.get("id"), "name": tc["name"], "args": tc.get("args", {})})
        if isinstance(m, ToolMessage):
            tool_results.append({"id": m.tool_call_id, "name": m.name, "content": m.content})

    final_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage) and not m.tool_calls), None)
    answer = final_ai.content if final_ai is not None else ""
    return {
        "question": question,
        "answer": answer,
        "tool_calls": tool_calls,
        "tool_results": tool_results,
        "messages": messages,
    }
