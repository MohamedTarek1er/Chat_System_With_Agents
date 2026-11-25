"""
graph/visualize.py – Render the LangGraph agent workflow to assets/agent_graph.png.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import core.llm as llm_module

# Provide fallback values so graph visualization works without live API keys in .env
if not llm_module.OPENAI_API_KEY:
    llm_module.OPENAI_API_KEY = "placeholder-key"
if not llm_module.OPENAI_MODEL:
    llm_module.OPENAI_MODEL = "gpt-4o-mini"

from graph.workflow import build_graph


def draw_graph(output_dir: Path | None = None) -> Path:
    """Build the workflow graph and save its diagram into assets/."""
    assets_dir = output_dir or (BASE_DIR / "assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    compiled = build_graph()
    drawable = compiled.get_graph()

    mermaid_path = assets_dir / "agent_graph.mmd"
    mermaid_path.write_text(drawable.draw_mermaid(), encoding="utf-8")

    png_path = assets_dir / "agent_graph.png"
    drawable.draw_mermaid_png(output_file_path=str(png_path))

    return png_path


if __name__ == "__main__":
    output_path = draw_graph()
    print(f"Graph saved to: {output_path}")
