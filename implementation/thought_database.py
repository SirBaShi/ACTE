"""
Thought Pool Database - Stores and retrieves heuristic ideas
"""

import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ThoughtDatabase:
    """Stores and retrieves heuristic ideas (Thought Pool)."""

    def __init__(self, filepath: str = "data/ideas.json"):
        self.filepath = filepath
        self.ideas: List[Dict] = self._load()
        logger.info(f"ThoughtDatabase loaded {len(self.ideas)} ideas from {filepath}")

    def _load(self) -> List[Dict]:
        """Load ideas from JSON file."""
        if not os.path.exists(self.filepath):
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            return []

        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load ideas file: {e}. Starting fresh.")
            return []

    def _save(self):
        """Save ideas to JSON file."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.ideas, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved {len(self.ideas)} ideas to {self.filepath}")

    def add_idea(self,
                 idea: str,
                 critique: str = None,
                 code: str = None,
                 score: Optional[float] = None,
                 metadata: Optional[Dict] = None) -> int:
        """Add a new idea to the pool."""
        idea_entry = {
            'id': len(self.ideas),
            'idea': idea,
            'critique': critique,
            'code': code,
            'score': score,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'iteration': len(self.ideas)
        }

        self.ideas.append(idea_entry)
        self._save()
        logger.info(f"Added idea #{idea_entry['id']} to Thought Pool")
        return idea_entry['id']

    def get_top_ideas(self, k: int = 3) -> Tuple[List[str], List[float]]:
        """Get top-k ideas by score."""
        # 确保返回 2 个值，即使为空
        if not self.ideas:
            return [], []

        sorted_ideas = sorted(
            self.ideas,
            key=lambda x: (x.get('score') is not None, x.get('score', 0)),
            reverse=True
        )

        top_k = sorted_ideas[:k]
        ideas = [item['idea'] for item in top_k]
        scores = [item.get('score') for item in top_k]

        return ideas, scores

    def update_score(self, idea_id: int, score: float):
        """Update score for an idea after evaluation."""
        for item in self.ideas:
            if item['id'] == idea_id:
                item['score'] = score
                item['metadata']['evaluated_at'] = datetime.now().isoformat()
                break
        self._save()
        logger.info(f"Updated idea #{idea_id} score to {score}")

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        if not self.ideas:
            return {'total': 0, 'evaluated': 0, 'avg_score': None}

        evaluated = [i for i in self.ideas if i.get('score') is not None]
        avg_score = sum(i['score'] for i in evaluated) / len(evaluated) if evaluated else None

        return {
            'total': len(self.ideas),
            'evaluated': len(evaluated),
            'avg_score': avg_score,
            'best_score': max((i['score'] for i in evaluated), default=None)
        }