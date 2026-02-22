"""
Software Engineering Toolkit for OpenSage.

From the paper: "OpenSage features a hierarchical, graph-based memory system
for efficient management and a specialized toolkit tailored to software
engineering tasks."

This toolkit provides tools for:
- File reading/writing
- Code execution
- Shell commands
- Code search and analysis
- Patch application
- Test execution
"""

from opensage.tools.se_toolkit.core import get_se_toolkit

__all__ = ["get_se_toolkit"]
