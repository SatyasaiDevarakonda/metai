"""AgentNotebook — durable scratchpad for long-horizon planning.

The notebook lives outside the LLM context window. The LLM writes to it
via structured directives (NOTE / RECALL / COMMIT / UPDATE_PLAN) embedded
in its Operating Brief. The 30-day long-horizon env truncates the prompt
so the LLM *cannot* solve the task without using the notebook — the only
way it remembers what it decided on day 3 by day 27 is by writing it down.
"""

from freshprice_env.notebook.agent_notebook import (
    AgentNotebook,
    Commitment,
    NotebookEntry,
)
from freshprice_env.notebook.notebook_directives import (
    NotebookDirective,
    NotebookDirectiveExecutor,
    NotebookDirectiveResult,
    extract_notebook_directives,
)

__all__ = [
    "AgentNotebook",
    "Commitment",
    "NotebookDirective",
    "NotebookDirectiveExecutor",
    "NotebookDirectiveResult",
    "NotebookEntry",
    "extract_notebook_directives",
]
