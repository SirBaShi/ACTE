"""
ATCE Demo Run Script - Strictly following Project Proposal (3 Agents)
Usage: python demo_run.py
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from implementation.llm_client import create_llm_client
from implementation.agents import Architect, DevilsAdvocate, Engineer
from implementation.thought_database import ThoughtDatabase


def setup_logging(log_file: str = "logs/atce_run.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def run_atce_demo(llm_provider: str = "openrouter",
                  model: str = '',
                  num_iterations: int = 3,
                  temperature: float = 0.7):
    logger = logging.getLogger(__name__)

    print("\n" + "=" * 70)
    print("  ADVERSARIAL THOUGHT-TO-CODE EVOLUTION (ATCE) - DEMO RUN")
    print("  Strictly following Project Proposal (3 Agents)")
    print("=" * 70)
    print(f"  Provider: {llm_provider}")
    print(f"  Model: {model or 'default'}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    logger.info(f"Initializing {llm_provider} client...")
    try:
        llm_kwargs = {'provider': llm_provider}
        if model:
            llm_kwargs['model'] = model
        llm_client = create_llm_client(**llm_kwargs)
        logger.info("LLM client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    thought_db = ThoughtDatabase(filepath="data/ideas.json")
    logger.info(f"Thought Database loaded ({len(thought_db.ideas)} existing ideas)")

    # Initialize 3 Agents as per Proposal
    architect = Architect(llm_client)
    critic = DevilsAdvocate(llm_client)
    engineer = Engineer(llm_client)

    results = []

    for i in range(num_iterations):
        print(f"\n{'=' * 70}")
        print(f"  ITERATION {i + 1}/{num_iterations}")
        print(f"{'=' * 70}")

        iteration_start = time.time()

        # --- Step 1: Architect ---
        print("\n🏗️  [STEP 1] ARCHITECT - Generating heuristic idea...")
        prev_ideas, prev_scores = thought_db.get_top_ideas(k=3)
        raw_idea = architect.generate_idea(prev_ideas, prev_scores)
        print(f"   Raw Idea:\n   {'-' * 65}\n   {raw_idea}\n   {'-' * 65}\n")

        # --- Step 2: Devil's Advocate ---
        print("😈 [STEP 2] DEVIL'S ADVOCATE - Critiquing idea...")
        critique = critic.critique(raw_idea)
        print(f"   Critique:\n   {'-' * 65}\n   {critique}\n   {'-' * 65}\n")

        # --- Step 3: Engineer (Includes Refinement Logic) ---
        # Note: Per Proposal, Engineer translates refined strategies by incorporating critique
        print("👨‍ [STEP 3] ENGINEER - Generating Robust Code (with Refinement)...")
        code = engineer.generate_code(raw_idea, critique)
        code_preview = code[:400] + ('...' if len(code) > 400 else '')
        print(f"   Generated Code:\n   {'-' * 65}\n   {code_preview}\n   {'-' * 65}\n")

        # --- Step 4: Save to Thought Pool ---
        # We store the Idea + Critique as the "Validated Thought"
        idea_id = thought_db.add_idea(
            idea=raw_idea,  # Store original idea
            critique=critique,  # Store critique as validation record
            code=code,
            score=None,
            metadata={
                'iteration': i + 1,
                'demo': True,
                'model': model or 'default',
                'duration_sec': time.time() - iteration_start
            }
        )
        print(f"💾 [STEP 4] Saved to Thought Pool (ID: {idea_id})")

        results.append({
            'iteration': i + 1,
            'idea_id': idea_id,
            'raw_idea': raw_idea,
            'critique': critique,
            'code': code,
            'duration_sec': time.time() - iteration_start
        })

        # Save iteration log
        iteration_log = f"logs/iteration_{i + 1}.txt"
        os.makedirs("logs", exist_ok=True)
        with open(iteration_log, 'w', encoding='utf-8') as f:
            f.write(f"ATCE Iteration {i + 1}\n")
            f.write(f"Model: {model or 'default'}\n")
            f.write(f"Duration: {time.time() - iteration_start:.2f}s\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"RAW IDEA:\n{raw_idea}\n\n")
            f.write(f"CRITIQUE:\n{critique}\n\n")
            f.write(f"CODE:\n{code}\n")
        print(f"📄 Iteration log saved to: {iteration_log}")

    # Print summary
    total_duration = sum(r['duration_sec'] for r in results)
    print(f"\n{'=' * 70}")
    print("  DEMO COMPLETE - SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total iterations: {num_iterations}")
    print(f"  Total duration: {total_duration:.2f}s")
    print(f"  Ideas in Thought Pool: {len(thought_db.ideas)}")
    print(f"{'=' * 70}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="ATCE Demo Run")
    parser.add_argument("--provider", type=str, default="openrouter", choices=["openrouter", "openai"])
    parser.add_argument("--model", type=str, default="openrouter/free")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--log-file", type=str, default="logs/atce_run.log")

    args = parser.parse_args()
    logger = setup_logging(args.log_file)
    logger.info(f"Starting ATCE demo with provider={args.provider}, model={args.model}, iterations={args.iterations}")

    try:
        results = run_atce_demo(
            llm_provider=args.provider,
            model=args.model,
            num_iterations=args.iterations,
            temperature=args.temperature
        )
        logger.info(f"Demo completed successfully with {len(results)} iterations")
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()