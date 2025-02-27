# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Optional

import tqdm

from openeqa.utils.openai_utils import (
    call_openai_api,
    prepare_openai_messages,
    set_openai_key,
)
from openeqa.utils.prompt_utils import load_prompt




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default="data/open-eqa-v0.json",
        help="path to EQA dataset (default: data/open-eqa-v0.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-11-20",
        help="GPT model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="gpt seed (default: 1234)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="gpt temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="gpt maximum tokens (default: 128)",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default="data/results",
        help="output directory (default: data/results)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="continue running on API errors (default: false)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="only process the first 5 questions",
    )
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    args.output_path = args.output_directory / (
        args.dataset.stem + "-{}-{}.json".format(args.model, args.seed)
    )
    return args

# original, cant deal with the situation where ":" is not after A
# def parse_output(output: str) -> str:
#     print(output)
#     start_idx = output.find("A:")
#     if start_idx == -1:
#         raise ValueError("Invalid output string: {}".format(output))
#     end_idx = output.find("\n", start_idx)
#     if end_idx == -1:
#         return output[start_idx:].replace("A:", "").strip()
    
#     return output[start_idx:end_idx].replace("A:", "").strip()

def parse_output(output: str) -> str:
    # 如果输出一开始既没有 "A" 也没有 "A:"，自动加上 "A: "
    if not output.startswith("A") and not output.startswith("A:"):
        output = f"A: {output.strip()}"

    # 找到第一个 "A" 的起始位置
    start_idx = output.find("A")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))

    # 提取从第一个 "A" 开始的内容
    output = output[start_idx:].strip()

    # 如果答案缺少冒号 ":", 自动补全
    if not output.startswith("A:"):
        output = f"A: {output[1:].strip()}"
    
    # 找到 "A:" 的实际起始位置
    start_idx = output.find("A:")
    if start_idx == -1:
        raise ValueError("Invalid output string: {}".format(output))

    # 舍去从第二个换行符开始的所有内容
    first_newline_idx = output.find("\n", start_idx)
    if first_newline_idx != -1:
        second_newline_idx = output.find("\n", first_newline_idx + 1)
        if second_newline_idx != -1:
            output = output[:second_newline_idx].strip()

    # 去掉 "A:" 并返回最终答案
    return output[start_idx:].replace("A:", "").strip()

    


    return output[start_idx:end_idx].replace("A:", "").strip()

    




def ask_question(
    question: str,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4o-2024-11-20",
    openai_seed: int = 1234,
    openai_max_tokens: int = 128,
    openai_temperature: float = 0.2,
    force: bool = False,
) -> Optional[str]:
    try:
        prompt = load_prompt("blind-llm")
        set_openai_key(key=openai_key)
        messages = prepare_openai_messages(prompt.format(question=question))
        output = call_openai_api(
            messages=messages,
            model=openai_model,
            seed=openai_seed,
            max_tokens=openai_max_tokens,
            temperature=openai_temperature,
        )
        print(output)
        return parse_output(output)
    except Exception as e:
        if not force:
            traceback.print_exc()
            raise e


def main(args: argparse.Namespace):
    # check for openai api key
    assert "OPENAI_API_KEY" in os.environ

    # load dataset
    dataset = json.load(args.dataset.open("r"))
    print("found {:,} questions".format(len(dataset)))

    # load results
    results = []
    if args.output_path.exists():
        results = json.load(args.output_path.open())
        print("found {:,} existing results".format(len(results)))
    completed = [item["question_id"] for item in results]

    # process data
    for idx, item in enumerate(tqdm.tqdm(dataset)):
        if args.dry_run and idx >= 5:
            break

        # skip completed questions
        question_id = item["question_id"]
        if question_id in completed:
            continue  # skip existing

        # generate answer
        question = item["question"]
        answer = ask_question(
            question=question,
            openai_model=args.model,
            openai_seed=args.seed,
            openai_max_tokens=args.max_tokens,
            openai_temperature=args.temperature,
            force=args.force,
        )

        # store results
        results.append({"question_id": question_id, "answer": answer})
        json.dump(results, args.output_path.open("w"), indent=2)

    # save at end (redundant)
    json.dump(results, args.output_path.open("w"), indent=2)
    print("saving {:,} answers".format(len(results)))


if __name__ == "__main__":
    main(parse_args())
