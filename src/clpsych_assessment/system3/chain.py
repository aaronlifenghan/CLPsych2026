"""
LLM chain construction for CLPsych 2026 tasks.

Supports multiple backends:
  - ollama:  Local models via Ollama (Llama, Gemma, Qwen, Phi, ...)
  - openai:  OpenAI API or any OpenAI-compatible server
  - google:  Google Gemini models via the Generative AI API
  - hf:      HuggingFace transformers (MentalLLaMA, PsyLLM, ...)

Each backend returns a LangChain-compatible model or a raw llm_fn.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Callable, Literal, Optional, Type

from pydantic import BaseModel
from pydantic.json_schema import JsonSchemaValue

logger = logging.getLogger(__name__)

Messages = list[dict]
LLMFn = Callable[[Messages], str]


# ── Model registry ───────────────────────────────────────────────

MODELS: dict[str, dict] = {
    # Ollama models (local)
    "llama3.1": {
        "backend": "ollama", "model_id": "llama3.1:8b",
        "description": "Llama 3.1 8B — general-purpose baseline",
    },
    "llama3.1:70b": {
        "backend": "ollama", "model_id": "llama3.1:70b",
        "description": "Llama 3.1 70B — strong, needs ~40GB VRAM",
    },
    "gemma2:9b": {
        "backend": "ollama", "model_id": "gemma2:9b",
        "description": "Gemma 2 9B — strong for its size",
    },
    "gemma2:27b": {
        "backend": "ollama", "model_id": "gemma2:27b",
        "description": "Gemma 2 27B — very strong, ~16GB VRAM",
    },
    "qwen2.5:7b": {
        "backend": "ollama", "model_id": "qwen2.5:7b",
        "description": "Qwen 2.5 7B — good structured output",
    },
    "qwen2.5:32b": {
        "backend": "ollama", "model_id": "qwen2.5:32b",
        "description": "Qwen 2.5 32B — very capable, ~20GB VRAM",
    },
    "phi3:14b": {
        "backend": "ollama", "model_id": "phi3:14b",
        "description": "Phi-3 14B — compact, strong reasoning",
    },
    "mistral:7b": {
        "backend": "ollama", "model_id": "mistral:7b",
        "description": "Mistral 7B — fast baseline",
    },
    "mistral-nemo": {
        "backend": "ollama", "model_id": "mistral-nemo",
        "description": "Mistral Nemo 12B — improved Mistral",
    },
    # Google Gemini (API)
    "gemini-flash": {
        "backend": "google", "model_id": "gemini-2.0-flash",
        "description": "Gemini 2.0 Flash — fast, good for experimentation",
    },
    "gemini-pro": {
        "backend": "google", "model_id": "gemini-2.5-pro-preview-06-05",
        "description": "Gemini 2.5 Pro — strongest Gemini",
    },
    # OpenAI (API)
    "gpt-4o-mini": {
        "backend": "openai", "model_id": "gpt-4o-mini",
        "description": "GPT-4o Mini — fast, cheap OpenAI",
    },
    "gpt-4o": {
        "backend": "openai", "model_id": "gpt-4o",
        "description": "GPT-4o — strong OpenAI model",
    },
    # HuggingFace mental health models
    "mentalllama": {
        "backend": "hf_causal", "model_id": "klyang/MentalLLaMA-chat-13B",
        "description": "MentalLLaMA 13B — mental health fine-tuned",
    },
    "mental-alpaca": {
        "backend": "hf_causal", "model_id": "NEU-HAI/mental-alpaca",
        "description": "Mental-Alpaca — mental health instruction-tuned",
    },
    "mental-flan-t5": {
        "backend": "hf_seq2seq", "model_id": "NEU-HAI/mental-flan-t5-large",
        "description": "Mental-FLAN-T5-large — seq2seq",
    },
    "psyllm": {
        "backend": "hf_causal", "model_id": "AIMH/SQPsychLLM-8b-gemma",
        "description": "PsyLLM 8B — psychology-focused Gemma fine-tune",
    },
}


# ── Backend: Ollama ──────────────────────────────────────────────

def get_ollama_model(
    model_name: str,
    base_url: str = "http://localhost:11434",
    format: Optional[JsonSchemaValue] = "",
    options: Optional[dict] = None,
):
    """Create a LangChain ChatOllama model."""
    from langchain_ollama import ChatOllama
    opts = options or {}
    return ChatOllama(
        base_url=base_url,
        model=model_name,
        format=format,
        num_ctx=opts.get("num_ctx", 32768),
        temperature=opts.get("temperature", 0.0),
        top_k=opts.get("top_k", 40),
        top_p=opts.get("top_p", 0.9),
    )


# ── Backend: OpenAI ──────────────────────────────────────────────

def get_openai_model(
    model_name: str,
    api_key: str = "",
    base_url: str = "",
    format: Optional[JsonSchemaValue] = None,
    options: Optional[dict] = None,
):
    """Create a LangChain ChatOpenAI model."""
    from langchain_openai import ChatOpenAI
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    model = ChatOpenAI(
        model=model_name,
        api_key=key,
        base_url=base_url or None,
        temperature=(options or {}).get("temperature", 0.0),
    )
    if format:
        return model.with_structured_output(schema=format, method="json_schema")
    return model


# ── Backend: Google Gemini ───────────────────────────────────────

def get_google_model(
    model_name: str,
    api_key: str = "",
    format: Optional[JsonSchemaValue] = None,
    options: Optional[dict] = None,
):
    """
    Create a LangChain ChatGoogleGenerativeAI model.

    Requires: pip install langchain-google-genai

    Set GOOGLE_API_KEY env var or pass api_key directly.
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "Google Gemini backend requires langchain-google-genai.\n"
            "Install: pip install langchain-google-genai"
        )

    key = api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        raise ValueError(
            "Google Gemini requires an API key. "
            "Set GOOGLE_API_KEY or pass --api-key."
        )

    opts = options or {}
    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=opts.get("temperature", 0.0),
        max_output_tokens=opts.get("max_tokens", 2048),
    )
    # Gemini structured output via with_structured_output
    if format and isinstance(format, dict):
        try:
            return model.with_structured_output(schema=format)
        except Exception as e:
            logger.warning(
                f"Gemini structured output setup failed ({e}), "
                f"falling back to text + parser"
            )
    return model


# ── Backend: HuggingFace (raw llm_fn, not LangChain) ────────────

def make_hf_causal_fn(
    model_name: str,
    device: str = "auto",
    temperature: float = 0.0,
    max_new_tokens: int = 1024,
    load_in_4bit: bool = False,
) -> LLMFn:
    """HuggingFace causal LM as a raw llm_fn."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    logger.info(f"Loading HF causal model: {model_name}")
    kwargs = {"device_map": device, "torch_dtype": torch.float16}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        except ImportError:
            logger.warning("bitsandbytes not available, full precision")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    def llm_fn(messages: Messages) -> str:
        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
            prompt = "\n\n".join(parts) + "\n\nAssistant:"
        out = pipe(
            prompt, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()
    return llm_fn


def make_hf_seq2seq_fn(
    model_name: str, device: str = "auto",
    temperature: float = 0.0, max_new_tokens: int = 1024,
) -> LLMFn:
    """HuggingFace seq2seq model as a raw llm_fn."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    logger.info(f"Loading HF seq2seq model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    def llm_fn(messages: Messages) -> str:
        parts = []
        for msg in messages:
            if msg["role"] == "system":
                parts.append(f"Instructions: {msg['content']}")
            elif msg["role"] == "user":
                parts.append(f"Input: {msg['content']}")
        prompt = "\n\n".join(parts) + "\n\nOutput:"
        out = pipe(
            prompt, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
        return out[0]["generated_text"].strip()
    return llm_fn


# ── Unified model constructor ────────────────────────────────────

def get_model(
    provider: str = "ollama",
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: str = "",
    format: Optional[JsonSchemaValue] = "",
    options: Optional[dict] = None,
):
    """Get a LangChain model. Returns None for HF backends."""
    if provider == "ollama":
        return get_ollama_model(model_name or "llama3.1:8b",
                                base_url=base_url or "http://localhost:11434",
                                format=format, options=options)
    elif provider == "openai":
        return get_openai_model(model_name or "gpt-4o-mini", api_key=api_key,
                                base_url=base_url or "",
                                format=format if isinstance(format, dict) else None,
                                options=options)
    elif provider == "google":
        return get_google_model(model_name or "gemini-2.0-flash", api_key=api_key,
                                format=format if isinstance(format, dict) else None,
                                options=options)
    elif provider in ("hf_causal", "hf_seq2seq"):
        return None  # use build_llm_fn instead
    else:
        raise ValueError(f"Unknown provider: {provider}")


def build_llm_fn(
    model_key: str, api_key: str = "", device: str = "auto",
    temperature: float = 0.0, max_tokens: int = 1024,
    load_in_4bit: bool = False, model_id_override: str = "",
) -> Optional[LLMFn]:
    """Build raw llm_fn for HF models. Returns None for non-HF backends."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODELS.keys())}")
    cfg = MODELS[model_key]
    backend = cfg["backend"]
    model_id = model_id_override or cfg["model_id"]

    if backend == "hf_causal":
        return make_hf_causal_fn(model_id, device=device, temperature=temperature,
                                 max_new_tokens=max_tokens, load_in_4bit=load_in_4bit)
    elif backend == "hf_seq2seq":
        return make_hf_seq2seq_fn(model_id, device=device, temperature=temperature,
                                  max_new_tokens=max_tokens)
    return None


# ── Chain construction ───────────────────────────────────────────

def create_chain(model, response_model: Type[BaseModel], prompt_name: str):
    """Create a LangChain chain with structured output parsing."""
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    parser = PydanticOutputParser(pydantic_object=response_model)
    template_path = Path(__file__).parent / "prompts" / f"{prompt_name}.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")
    prompt = ChatPromptTemplate.from_template(template).partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | model | parser


def create_raw_chain(
    llm_fn: LLMFn,
    response_model: Type[BaseModel],
    prompt_name: str,
):
    """
    Chain using a raw llm_fn (for HF models without LangChain).

    Returns a callable with .invoke({post_text: str}) → Pydantic model.
    Includes retry logic for parse failures.
    """
    from langchain_core.output_parsers import PydanticOutputParser

    parser = PydanticOutputParser(pydantic_object=response_model)
    template_path = Path(__file__).parent / "prompts" / f"{prompt_name}.md"
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt not found: {template_path}")

    template = template_path.read_text(encoding="utf-8")
    format_instructions = parser.get_format_instructions()

    class RawChain:
        def invoke(self, inputs: dict):
            prompt_text = template.replace("{post_text}", inputs.get("post_text", ""))
            prompt_text = prompt_text.replace("{format_instructions}", format_instructions)
            messages = [
                {"role": "system", "content": "You are an expert psychologist."},
                {"role": "user", "content": prompt_text},
            ]
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    raw = llm_fn(messages)
                    return parser.parse(raw)
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Parse attempt {attempt+1} failed: {e}")
                    else:
                        logger.error(f"All parse attempts failed: {e}")
                        raise
    return RawChain()


def list_available_models():
    """Print all available models."""
    print("\nAvailable models:\n")
    for key, cfg in MODELS.items():
        print(f"  {key:<20s}  [{cfg['backend']:10s}]  {cfg['description']}")
    print()