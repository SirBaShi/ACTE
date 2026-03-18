# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Class for sampling new programs."""
from __future__ import annotations
from abc import ABC, abstractmethod

from typing import Collection, Sequence, Type
import numpy as np
import time

from implementation import evaluator
from implementation import programs_database

from agents import Architect, DevilsAdvocate, Engineer
from thought_database import ThoughtDatabase


class LLM(ABC):
    """Language model that predicts continuation of provided source code.

    RZ: The sampled function code must be trimmed! Especially using instruct-based LLM.
    -For example, the sampled function code (with description) is:
    ------------------------------------------------------------------------------------------------------------------
    Here is the function.
    def priority_v2(..., ...) -> Any:
        a = np.array([1, 2, 3])
        if len(a) > 2:
            return a / a.sum()
        else:
            return a / a.mean()
    This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------
    -The descriptions above the function's signature, and the function's signature must be removed.
    -The above code must be trimmed as follows:
    ------------------------------------------------------------------------------------------------------------------
        a = np.array([1, 2, 3])
            if len(a) > 2:
                return a / a.sum()
            else:
                return a / a.mean()
        Here is the function. This function is going to ..., and returns ...[Descriptions by LLM]
    ------------------------------------------------------------------------------------------------------------------
    Please note that the indent must be preserved. And the additional descriptions can also be preserved,
    which will be trimmed by Evaluator.
    """

    def __init__(self, samples_per_prompt: int) -> None:
        self._samples_per_prompt = samples_per_prompt

    def _draw_sample(self, prompt: str) -> str:
        """Returns a predicted continuation of `prompt`."""
        raise NotImplementedError('Must provide a language model.')

    @abstractmethod
    def draw_samples(self, prompt: str) -> Collection[str]:
        """Returns multiple predicted continuations of `prompt`."""
        return [self._draw_sample(prompt) for _ in range(self._samples_per_prompt)]


class Sampler:
    """Node that samples program continuations and sends them for analysis.
    """
    _global_samples_nums: int = 1  # RZ: this variable records the global sample nums

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            max_sample_nums: int | None = None,
            llm_class: Type[LLM] = LLM
    ):
        self._samples_per_prompt = samples_per_prompt
        self._database = database
        self._evaluators = evaluators
        self._llm = llm_class(samples_per_prompt)
        self._max_sample_nums = max_sample_nums

    def sample(self, **kwargs):
        """Continuously gets prompts, samples programs, sends them for analysis.
        """
        while True:
            # stop the search process if hit global max sample nums
            if self._max_sample_nums and self.__class__._global_samples_nums >= self._max_sample_nums:
                break
            try:
                prompt = self._database.get_prompt()
                reset_time = time.time()
                samples = self._llm.draw_samples(prompt.code)
                sample_time = (time.time() - reset_time) / self._samples_per_prompt
                # This loop can be executed in parallel on remote evaluator machines.
                for sample in samples:
                    self._global_sample_nums_plus_one()  # RZ: add _global_sample_nums
                    cur_global_sample_nums = self._get_global_sample_nums()
                    chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                    chosen_evaluator.analyse(
                        sample,
                        prompt.island_id,
                        prompt.version_generated,
                        **kwargs,
                        global_sample_nums=cur_global_sample_nums,
                        sample_time=sample_time
                    )
            except:
                continue

    def _get_global_sample_nums(self) -> int:
        return self.__class__._global_samples_nums

    def set_global_sample_nums(self, num):
        self.__class__._global_samples_nums = num

    def _global_sample_nums_plus_one(self):
        self.__class__._global_samples_nums += 1



class ATCESampler(Sampler):
    """Sampler with Adversarial Thought-to-Code Evolution."""

    def __init__(
            self,
            database: programs_database.ProgramsDatabase,
            evaluators: Sequence[evaluator.Evaluator],
            samples_per_prompt: int,
            llm_client,  # 新增：LLM 客户端
            thought_database: ThoughtDatabase,  # 新增：Thought Pool
            max_sample_nums: int | None = None,
    ):
        super().__init__(database, evaluators, samples_per_prompt, max_sample_nums)

        # 初始化三个 Agent
        self.architect = Architect(llm_client)
        self.critic = DevilsAdvocate(llm_client)
        self.engineer = Engineer(llm_client)

        self.thought_db = thought_database
        self._enable_atce = True  # 可开关，方便对比实验

    def sample(self, **kwargs):
        """Modified sampling loop with ATCE."""
        while True:
            if self._max_sample_nums and self._global_samples_nums >= self._max_sample_nums:
                break
            try:
                # ========== ATCE 核心流程 ==========
                if self._enable_atce:
                    # Step 1: Architect 生成 Idea
                    previous_ideas = self.thought_db.get_top_ideas(k=3)
                    raw_idea = self.architect.generate(previous_ideas)

                    # Step 2: Devil's Advocate 批评 (纯文本，快速)
                    critique = self.critic.generate(raw_idea)

                    # Step 3: 根据批评修正 Idea (可选：可让 Architect 再 refinement)
                    refined_idea = self._refine_idea(raw_idea, critique)

                    # Step 4: Engineer 生成代码
                    prompt_code = self._database.get_prompt()
                    sample = self.engineer.generate(refined_idea, critique)

                    # Step 5: 保存 Thought 到 Pool (用于下一轮进化)
                    self.thought_db.add_idea(refined_idea, score=None)  # score 后续更新
                else:
                    # 原始 FunSearch 流程 (用于 baseline 对比)
                    prompt = self._database.get_prompt()
                    samples = self._llm.draw_samples(prompt.code)
                    sample = samples[0]

                # ========== 评估 ==========
                reset_time = time.time()
                sample_time = (time.time() - reset_time) / self._samples_per_prompt

                self._global_sample_nums_plus_one()
                cur_global_sample_nums = self._get_global_sample_nums()

                chosen_evaluator: evaluator.Evaluator = np.random.choice(self._evaluators)
                chosen_evaluator.analyse(
                    sample,
                    prompt.island_id if hasattr(prompt, 'island_id') else 0,
                    prompt.version_generated if hasattr(prompt, 'version_generated') else 0,
                    **kwargs,
                    global_sample_nums=cur_global_sample_nums,
                    sample_time=sample_time,
                    atce_log={  # 新增：记录 ATCE 日志用于 Report
                        'idea': raw_idea if self._enable_atce else None,
                        'critique': critique if self._enable_atce else None,
                        'refined_idea': refined_idea if self._enable_atce else None,
                    }
                )
            except Exception as e:
                print(f"Sampling error: {e}")
                continue

    def _refine_idea(self, idea: str, critique: str) -> str:
        """让 Architect 根据批评修正 Idea."""
        prompt = f"""Original idea: {idea}
        Critique: {critique}

        Refine the idea to address these concerns. Output ONLY the refined strategy."""
        return self.architect.llm.generate(self.architect.system_prompt, prompt)