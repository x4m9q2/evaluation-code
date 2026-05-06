import copy
import datetime
import json
import os
import re

from pathlib import Path
from typing import List

try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
from tqdm import tqdm

from dataset_loader import DatasetLoader
from judge_prompt import PROMPT
from model import Model


class Evaluator:
    """
    Evaluator class for processing evaluation tasks using the xVerify model.
    """

    def __init__(
        self,
        model: Model,
        process_num: int = 1,
        batch_size: int = 8
    ):
        self.model = copy.deepcopy(model)
        self.model_name = model.model_name
        self.process_num = process_num
        self.batch_size = batch_size
        self.prompt = PROMPT

        if self.batch_size <= 0:
            raise ValueError("batch_size should be greater than 0")
        if process_num != 1:
            logger.warning(
                "process_num is ignored in the current single-thread batch mode. "
                "Please tune batch_size instead."
            )

    def load_data(self, data_path: str, data_size: int = None) -> List[dict]:
        data_size = data_size if data_size is not None else -1
        return DatasetLoader.fixed_load(data_path, data_size)

    def construct_prompt(self, data: List[dict]) -> None:
        for item in data:
            user_input = self.prompt.format(
                question=item['question'],
                output=item['llm_output'],
                answer=item['correct_answer']
            )
            item['prompt'] = user_input

    def gen(self, data_point: dict) -> dict:
        result = self.model.request(data_point['prompt'])
        data_point[f'{self.model_name}_judgment_result'] = result
        return data_point

    def batch_gen(self, data: List[dict], data_name: str) -> List[dict]:
        results: List[dict] = []
        for start_idx in tqdm(
            range(0, len(data), self.batch_size),
            total=(len(data) + self.batch_size - 1) // self.batch_size,
            desc=f'{self.model_name}_{data_name}'
        ):
            batch = data[start_idx:start_idx + self.batch_size]
            prompts = [item['prompt'] for item in batch]
            batch_results = self.model.request_batch(prompts)
            for item, result in zip(batch, batch_results):
                item[f'{self.model_name}_judgment_result'] = result
                results.append(item)
        return results

    def stat_results(self, results: List[dict]) -> dict:
        valid_num = 0
        correct_num = 0
        incorrect_num = 0
        for item in results:
            raw_label = str(item[f'{self.model_name}_judgment_result'])
            item_label = raw_label.strip().lower()

            matches = re.findall(r'\b(correct|incorrect)\b', item_label)
            if matches:
                item_label = matches[-1]
                valid_num += 1
                item['judge_valid'] = 'True'
            else:
                item['judge_valid'] = 'False'

            if item_label == 'correct':
                correct_num += 1
            elif item_label == 'incorrect':
                incorrect_num += 1

        return {
            "Valid_num": valid_num,
            "Correct_num": correct_num,
            "Incorrect_num": incorrect_num,
            "Accuracy": correct_num / len(results)
        }

    def save_output(self, output: dict) -> None:
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=4)
        logger.info(f"Output saved to {self.output_path}!")

    def single_evaluate(self, question: str, llm_output: str, correct_answer: str) -> str:
        data_point = {
            'question': question,
            'llm_output': llm_output,
            'correct_answer': correct_answer
        }

        user_input = self.prompt.format(
            question=data_point['question'],
            output=data_point['llm_output'],
            answer=data_point['correct_answer']
        )
        data_point['prompt'] = user_input

        result = self.gen(data_point)
        return result[f'{self.model_name}_judgment_result']

    def evaluate(self, data_path: str, output_path: str, data_size: int = None) -> dict:
        data = self.load_data(data_path, data_size)
        data_name = Path(data_path).stem
        data_size = len(data)

        info = {
            'llm': {
                "model_name": self.model_name,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "top_p": self.model.top_p
            },
            'dataset': data_name,
            'data_num': data_size,
            'datetime': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

        self.construct_prompt(data)
        results = self.batch_gen(data, data_name)
        stat_info = self.stat_results(results)

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'Eval_Judge_{self.model_name}_{data_name}_{data_size}_{timestamp}.json'
        self.output_path = os.path.join(output_path, output_name)

        self.save_output({
            'info': info,
            'stat_info': stat_info,
            'results': results
        })

        return stat_info
