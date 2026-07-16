"""Local LangGraph Studio entry point for the live Nebius-backed agent.

``langgraph dev`` supplies its own in-memory persistence, so Studio can inspect
per-thread state and replay a local run without a graph-level checkpointer.
"""

from data_analysis_agent.config import load_settings
from data_analysis_agent.graph import build_graph
from data_analysis_agent.models import NebiusRoleModel
from data_analysis_agent.nebius_client import create_nebius_client
from data_analysis_agent.python_runner import LocalPythonRunner
from data_analysis_agent.studio_inputs import StudioInput, prepare_studio_input

settings = load_settings()
client = create_nebius_client(settings)
model = NebiusRoleModel(client=client, model=settings.nebius_model)

graph = build_graph(
    model=model,
    runner=LocalPythonRunner(),
    input_schema=StudioInput,
    input_adapter=prepare_studio_input,
)
