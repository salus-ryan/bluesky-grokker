"""Swarm – multi-model reasoning pipeline inspired by MOTL (Machine-Optimized
Thought Language) from elevate-foundry/braille.

Architecture:
  [Query] → ModelRouter → Distiller → SemanticCodec → [Compressed Knowledge]
                                ↓
                         SwarmPipeline (orchestrator)
"""

from swarm.router import ModelRouter
from swarm.codec import SemanticCodec
from swarm.distiller import Distiller
from swarm.pipeline import SwarmPipeline
from swarm.memory import BrailleMemory

__all__ = ["ModelRouter", "SemanticCodec", "Distiller", "SwarmPipeline", "BrailleMemory"]
