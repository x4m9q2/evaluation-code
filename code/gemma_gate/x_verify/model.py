import os

from typing import List, Literal, Tuple

import torch
try:
    from loguru import logger
except ModuleNotFoundError:
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
try:
    from openai import OpenAI
    from openai._exceptions import APITimeoutError
except (ModuleNotFoundError, ImportError):
    OpenAI = None

    class APITimeoutError(Exception):
        pass

# DeepSpeed/Transformers stacks in this env may expect torch.amp.custom_fwd/custom_bwd.
if hasattr(torch, "amp") and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
    if not hasattr(torch.amp, "custom_fwd") and hasattr(torch.cuda.amp, "custom_fwd"):
        def _custom_fwd(*args, **kwargs):
            kwargs.pop("device_type", None)
            return torch.cuda.amp.custom_fwd(*args, **kwargs)

        torch.amp.custom_fwd = _custom_fwd

    if not hasattr(torch.amp, "custom_bwd") and hasattr(torch.cuda.amp, "custom_bwd"):
        def _custom_bwd(*args, **kwargs):
            kwargs.pop("device_type", None)
            return torch.cuda.amp.custom_bwd(*args, **kwargs)

        torch.amp.custom_bwd = _custom_bwd

from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from tenacity import retry, stop_after_attempt, wait_random_exponential
except ModuleNotFoundError:
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_random_exponential(*args, **kwargs):
        return None

from base_template import BASE_TEMPLATE


RETRY_TIMES = 30
WAIT_TIME_UPPER = 30
WAIT_TIME_LOWER = 10
TIMEOUT = 60


class Model:
    """
    A class to interact with a xVerify model, supporting both local and API-based inference.
    """

    def __init__(
        self,
        model_name: str,
        model_path_or_url: str,
        inference_mode: Literal["api", "local"],
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.7
    ):
        if inference_mode not in ["api", "local"]:
            raise ValueError("inference_mode must be either 'local' or 'api'")

        if not (0 <= temperature <= 1):
            raise ValueError("temperature should be between 0 and 1")

        if max_tokens <= 0:
            raise ValueError("max_tokens should be greater than 0")

        self.model_name = model_name
        self.inference_mode = inference_mode
        self.model_path_or_url = self._normalize_local_model_path(model_path_or_url)
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self._local_tokenizer = None
        self._local_model = None
        self._local_device = None

    def _normalize_local_model_path(self, model_path_or_url: str) -> str:
        if self.inference_mode == "local" and os.path.isfile(model_path_or_url):
            return os.path.dirname(model_path_or_url)
        return model_path_or_url

    def _load_template(self) -> str:
        try:
            return BASE_TEMPLATE[self.model_name]
        except KeyError:
            logger.error(f"Base template for model '{self.model_name}' does not exist.")
            raise KeyError(f"Missing template for model '{self.model_name}'")
        except Exception:
            logger.exception("Unexpected error while loading the template")
            raise

    def _initialize_local_model(self) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
        if self._local_tokenizer is not None and self._local_model is not None:
            return self._local_tokenizer, self._local_model

        if not os.path.exists(self.model_path_or_url):
            logger.info(
                f"Model not found locally. Downloading model {self.model_name} from Huggingface."
            )
            os.system(
                f'huggingface-cli download --resume-download IAAR-Shanghai/{self.model_name} '
                f'--local-dir {self.model_path_or_url}'
            )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path_or_url, use_fast=False, trust_remote_code=True
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path_or_url,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        model.to(device)
        model.eval()

        self._local_tokenizer = tokenizer
        self._local_model = model
        self._local_device = device

        return self._local_tokenizer, self._local_model

    def _request_local_batch(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return []

        base_template = self._load_template()
        formatted_prompts = [base_template.format(query=prompt) for prompt in prompts]
        tokenizer, model = self._initialize_local_model()

        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self._local_device or model.device)

        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_tokens,
            temperature=self.temperature
        )

        prompt_len = inputs["input_ids"].shape[1]
        responses = []
        for sequence in output_ids:
            generated_ids = sequence[prompt_len:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            responses.append(response.strip())
        return responses

    def _request_local(self, prompt: str) -> str:
        return self._request_local_batch([prompt])[0]

    @retry(
        wait=wait_random_exponential(min=WAIT_TIME_LOWER, max=WAIT_TIME_UPPER),
        stop=stop_after_attempt(RETRY_TIMES),
        reraise=True
    )
    def _request_api(self, prompt: str) -> str:
        try:
            if OpenAI is None:
                raise ModuleNotFoundError("openai package is required for API inference mode")
            model = OpenAI(
                base_url=self.model_path_or_url,
                api_key=self.api_key
            )

            response_obj = model.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                timeout=TIMEOUT
            )

            return response_obj.choices[0].message.content
        except APITimeoutError as e:
            logger.warning(f"Request timed out: {repr(e)}")
            return ''
        except Exception as e:
            logger.warning(repr(e))
            raise

    def request(self, prompt: str) -> str:
        return self.request_batch([prompt])[0]

    def request_batch(self, prompts: List[str]) -> List[str]:
        if self.inference_mode == 'api':
            return [self._request_api(prompt) for prompt in prompts]
        return self._request_local_batch(prompts)
