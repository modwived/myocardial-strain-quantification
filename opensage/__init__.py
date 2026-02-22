"""
OpenSage: Self-programming Agent Generation Engine

Based on: "OpenSage: Self-programming Agent Generation Engine"
arXiv:2602.16891 - Hongwei Li et al., February 2026

The first ADK (Agent Development Kit) that enables LLMs to automatically
create agents with self-generated topology and toolsets while providing
comprehensive and structured memory support.

Key innovations:
- Self-Generated Agent Topology (vertical + horizontal)
- AI-driven Tool Construction & Container-based Execution
- Hierarchical Graph-Based Memory System
"""

from opensage.core.engine import OpenSage
from opensage.core.agent import SageAgent

__version__ = "0.1.0"
__all__ = ["OpenSage", "SageAgent"]
