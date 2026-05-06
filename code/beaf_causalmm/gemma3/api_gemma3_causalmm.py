import os
from pathlib import Path
from typing import Literal, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from causalmm_gemma3 import CausalMMGemma3


BUNDLE_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = str(BUNDLE_ROOT / "models/Gemma-3-4B-IT")
DTYPE = os.getenv("TORCH_DTYPE", "bfloat16")

app = FastAPI(title="Gemma 3 4B CausalMM API", version="1.0.0")
runner: Optional[CausalMMGemma3] = None


class GenerateRequest(BaseModel):
    prompt: str
    image_path: Optional[str] = None
    system: Optional[str] = None
    max_new_tokens: int = Field(default=128, ge=1, le=1024)
    gamma: float = Field(default=1.0, ge=0.0)
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    cf_mode: Literal["language", "vision", "both"] = "language"
    attention_method: Literal["reverse", "reverse_and_normalize", "random", "uniform", "shuffle", "none"] = (
        "reverse_and_normalize"
    )
    vision_method: Literal["shuffle", "uniform", "reverse", "random", "none"] = "shuffle"


class GenerateResponse(BaseModel):
    model: str
    text: str
    prompt_tokens: int
    completion_tokens: int


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


@app.on_event("startup")
def startup():
    global runner
    runner = CausalMMGemma3(model_path=MODEL_PATH, torch_dtype=_dtype_from_name(DTYPE))


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    assert runner is not None
    result = runner.generate(
        prompt=request.prompt,
        image_path=request.image_path,
        system=request.system,
        max_new_tokens=request.max_new_tokens,
        gamma=request.gamma,
        epsilon=request.epsilon,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        cf_mode=request.cf_mode,
        attention_method=request.attention_method,
        vision_method=request.vision_method,
    )
    return GenerateResponse(
        model=MODEL_PATH,
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
