"""
ATCE Agent Implementations - Strictly following Project Proposal (3 Agents)
Agents: Architect, Devil's Advocate, Engineer
"""

import logging
from typing import List, Optional
# from llm_client import LLMClient

logger = logging.getLogger(__name__)

# ========== System Prompts ==========

ARCHITECT_SYSTEM_PROMPT = """
You are an Algorithm Architect specializing in Online Bin Packing heuristics.
Your task is to design high-level strategies (NOT code) for packing items into bins.

Guidelines:
1. Think creatively about combining existing heuristics (Best Fit, First Fit, Worst Fit, etc.)
2. Consider dynamic thresholds, adaptive behaviors, and state-aware decisions
3. Be specific about conditions and actions
4. Output ONLY the strategy description, no code, no markdown

Example output:
"Use Best Fit when item size > 0.5 and bin utilization < 70%, otherwise use First Fit 
with a dynamic threshold based on the average item size seen so far."
"""

CRITIC_SYSTEM_PROMPT = """
You are the Devil's Advocate - a harsh algorithm critic for bin packing heuristics.
Your task is to find logical flaws, edge cases, and failure scenarios in proposed strategies.

Guidelines:
1. Be specific and constructive - identify exact failure conditions
2. Consider worst-case item sequences (descending, ascending, uniform, etc.)
3. Think about boundary conditions (empty bins, full bins, exact thresholds)
4. Floating-point comparison issues
5. Performance concerns
6. Output ONLY the critique, no code, no pleasantries

Example output:
"Flaw 1: If all items are exactly 0.5, the threshold comparison may fail due to floating point.
Fix: Use >= instead of >.
Flaw 2: When bins are nearly full, this strategy forces new bins unnecessarily.
Fix: Add a fallback to Best Fit when no bin meets the threshold."
"""

ENGINEER_SYSTEM_PROMPT = """
You are a Senior Python Engineer implementing bin packing heuristics.
Your task is to convert strategy descriptions into robust, production-ready Python code.

Guidelines:
1. Output ONLY the function body (no markdown, no explanations)
2. Include defensive logic to handle edge cases identified by the critic
3. Use numpy for efficient array operations
4. Add clear comments explaining key decisions
5. Ensure the code is syntactically correct and ready for execution

Function signature (DO NOT change):
def priority(current_item: float, bin_capacities: list, bin_loads: list) -> int:
    # Return the index of the chosen bin
"""


class ATCEAgent:
    """Base class for ATCE agents."""

    def __init__(self, llm_client, system_prompt: str, agent_name: str):
        self.llm = llm_client
        self.system_prompt = system_prompt
        self.agent_name = agent_name

    def generate(self, user_prompt: str, temperature: float = 0.7) -> str:
        logger.info(f"[{self.agent_name}] Generating response...")
        response = self.llm.generate(
            system_prompt=self.system_prompt,
            user_prompt=user_prompt,
            temperature=temperature
        )
        logger.info(f"[{self.agent_name}] Response generated ({len(response)} chars)")
        return response


class Architect(ATCEAgent):
    """
    Step 1: Generates high-level heuristic ideas (Thought).
    Corresponds to the "Thought Pool" evolution in ATCE.
    """

    def __init__(self, llm_client):
        super().__init__(llm_client, ARCHITECT_SYSTEM_PROMPT, "ARCHITECT")

    def generate_idea(self,
                      previous_ideas: List[str],
                      previous_scores: Optional[List[float]] = None) -> str:
        if previous_ideas:
            ideas_text = ""
            for i, idea in enumerate(previous_ideas):
                score_info = f" (Score: {previous_scores[i]:.4f})" if previous_scores else ""
                ideas_text += f"{i + 1}. {idea}{score_info}\n"

            user_prompt = f"""Previous successful heuristic strategies:
{ideas_text}

Propose a NEW strategy by:
1. Combining elements from multiple previous ideas
2. Modifying thresholds or conditions
3. Adding new adaptive behaviors

Output ONLY the new strategy description."""
        else:
            user_prompt = """Propose an initial heuristic strategy for Online Bin Packing.
Consider combining Best Fit, First Fit, or Worst Fit with adaptive thresholds.
Output ONLY the strategy description."""

        return self.generate(user_prompt)


class DevilsAdvocate(ATCEAgent):
    """
    Step 2: Critiques ideas before code generation (Text-level Attack).
    This is the KEY innovation of ATCE - adversarial validation at thought level.
    """

    def __init__(self, llm_client):
        super().__init__(llm_client, CRITIC_SYSTEM_PROMPT, "DEVILS_ADVOCATE")

    def critique(self, idea: str) -> str:
        user_prompt = f"""Analyze this heuristic strategy for Online Bin Packing:

"{idea}"

Identify:
1. Logical gaps or ambiguities
2. Edge cases where this fails (specific item sequences)
3. Boundary conditions (empty bins, full bins, threshold edges)
4. Floating-point comparison issues
5. Performance concerns

For each flaw, suggest a specific fix.
Be harsh but constructive."""

        return self.generate(user_prompt)


class Engineer(ATCEAgent):
    """
    Step 3: Converts refined ideas into robust Python code.
    Incorporates the Critic's feedback directly into the code implementation.
    """

    def __init__(self, llm_client):
        super().__init__(llm_client, ENGINEER_SYSTEM_PROMPT, "ENGINEER")

    def generate_code(self, idea: str, critique: str) -> str:
        """
        Generate Python code implementing the strategy.
        The refinement logic is handled here by incorporating the critique.
        """
        user_prompt = f"""Strategy to implement:
{idea}

Critic's concerns (MUST address these in your code):
{critique}

Write Python code that:
1. Implements the strategy correctly
2. Includes defensive logic for ALL identified edge cases
3. Handles floating-point comparisons safely (use epsilon)
4. Has fallback behaviors for edge cases

Output ONLY the function body (no markdown, no function signature)."""

        return self.generate(user_prompt, temperature=0.3)  # Lower temp for code stability


# ========== Agent Factory ==========

def create_atce_agents(llm_client) -> dict:
    """Create the 3 ATCE agents as per Project Proposal."""
    return {
        'architect': Architect(llm_client),
        'critic': DevilsAdvocate(llm_client),
        'engineer': Engineer(llm_client)
    }
