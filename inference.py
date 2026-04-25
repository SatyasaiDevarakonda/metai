"""Inference runner — judges run this to evaluate the submission.

Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables.
Emits [START], [STEP], [END] structured logs to stdout.
Uses OpenAI client for LLM calls.
"""

from __future__ import annotations

import json
import os
import time

from openai import OpenAI

from freshprice_env.enums import CurriculumScenario
from freshprice_env.freshprice_env import FreshPriceEnv
from freshprice_env.models import TaskGraderResult
from eval.task_graders import TaskGrader

# Load from environment variables — never hardcode
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or os.environ.get("OPENAI_API_KEY", ""),
)


def call_llm(prompt: str) -> str:
    """Call the LLM via OpenAI client. Returns the brief text."""
    # Split prompt into system (first section) and user (rest)
    parts = prompt.split("\n\n", 1)
    system_msg = parts[0] if len(parts) > 0 else ""
    user_msg = parts[1] if len(parts) > 1 else prompt

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=800,
        temperature=0.7,
    )
    return response.choices[0].message.content


def run_episode(
    scenario_name: str = "STABLE_WEEK",
    seed: int = 42,
    max_steps: int = 84,
) -> dict:
    """Run one complete episode and return results."""

    scenario = CurriculumScenario[scenario_name]

    class LLMClient:
        def generate(self, prompt: str) -> str:
            return call_llm(prompt)

    env = FreshPriceEnv(scenario=scenario, seed=seed, llm_client=LLMClient())

    # [START] log
    start_payload = {
        "episode_id": f"{scenario_name}_{seed}",
        "scenario": scenario_name,
        "seed": seed,
        "model": MODEL_NAME,
        "timestamp": time.time(),
    }
    print(f"[START] {json.dumps(start_payload)}")

    obs_str, info = env.reset()
    env_state = env.state()

    step_num = 0
    total_reward = 0.0
    done = False

    while not done and step_num < max_steps:
        # Generate brief via LLM
        brief = call_llm(obs_str)

        # Step the environment
        next_obs, reward, done, truncated, info = env.step(brief)
        env_state = env.state()
        total_reward += reward

        # [STEP] log
        step_payload = {
            "step": step_num,
            "tick": env_state.get("tick", 0),
            "reward": round(reward, 4),
            "wrr": round(env_state.get("wrr_so_far", 0.0), 4),
            "engine_type": env_state.get("engine_type", "PRICING"),
            "critical_batches": env_state.get("critical_batches", 0),
            "done": done,
        }
        print(f"[STEP] {json.dumps(step_payload)}")

        obs_str = next_obs
        step_num += 1

    final_reward = info.get("final_reward", {})

    # [END] log
    end_payload = {
        "episode_id": f"{scenario_name}_{seed}",
        "steps_completed": step_num,
        "total_reward": round(total_reward, 4),
        "final_wrr": round(final_reward.get("wrr", 0.0), 4),
        "brief_quality_score": round(final_reward.get("brief_quality_score", 0.0), 4),
        "anti_hack_violations": final_reward.get("anti_hack_violations", 0),
        "episode_valid": final_reward.get("episode_valid", False),
    }
    print(f"[END] {json.dumps(end_payload)}")

    return {**end_payload, "env": env}


def run_task_graders(seed: int = 42) -> None:
    """Run all 3 task graders and print scores."""
    print("\n" + "=" * 60)
    print("TASK GRADER EVALUATION")
    print("=" * 60)

    results = TaskGrader.run_all_tasks(
        model_fn=call_llm,
        seed=seed,
    )

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[TASK] {json.dumps({'task_id': result.task_id, 'task_name': result.task_name, 'score': round(result.score, 4), 'status': status})}")

    mean_score = sum(r.score for r in results) / len(results) if results else 0.0
    print(f"[SUMMARY] {json.dumps({'mean_task_score': round(mean_score, 4), 'tasks_passed': sum(1 for r in results if r.passed), 'tasks_total': len(results)})}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="QStorePrice AI Inference Runner")
    parser.add_argument("--scenario", default="STABLE_WEEK")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tasks-only", action="store_true",
                        help="Run only task graders, skip episode")
    parser.add_argument("--episodes-only", action="store_true",
                        help="Run only episode, skip task graders")
    args = parser.parse_args()

    if not args.tasks_only:
        run_episode(scenario_name=args.scenario, seed=args.seed)

    if not args.episodes_only:
        run_task_graders(seed=args.seed)
