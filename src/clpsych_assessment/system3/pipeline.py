"""
CLPsych 2026 — Generic pipeline for all tasks.

Supports:
  - Multiple LLM backends (Ollama, OpenAI, Gemini, HuggingFace)
  - Zero-shot and few-shot prompts
  - Retry logic for parse failures
  - Context window configuration
"""

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel

from .chain import (
    MODELS,
    create_chain,
    create_raw_chain,
    get_model,
    build_llm_fn,
)
from .preprocessor import format_timeline_for_llm, load_timeline_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class CLPsychPipeline:
    """Generic pipeline for all CLPsych 2026 tasks."""

    def __init__(
        self,
        response_model: Type[BaseModel],
        prompt_name: str,
        model_key: str = "llama3.1",
        config_path: Optional[str] = None,
        api_key: str = "",
        base_url: str = "",
        device: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        load_in_4bit: bool = False,
        fewshot: bool = False,
    ):
        if config_path is None:
            config_path = str(Path(__file__).parent / "config.yaml")

        self.config = self._load_config(config_path)
        self.response_model = response_model
        self.model_key = model_key
        self.fewshot = fewshot

        # Resolve prompt name (append _fewshot if requested)
        if fewshot and not prompt_name.endswith("_fewshot"):
            fewshot_name = f"{prompt_name}_fewshot"
            fewshot_path = Path(__file__).parent / "prompts" / f"{fewshot_name}.md"
            if fewshot_path.exists():
                prompt_name = fewshot_name
                logger.info(f"Using few-shot prompt: {prompt_name}")
            else:
                logger.warning(
                    f"Few-shot prompt {fewshot_name} not found, "
                    f"using zero-shot: {prompt_name}"
                )

        self.prompt_name = prompt_name
        self._setup_chain(api_key, base_url, device, temperature, max_tokens, load_in_4bit)

    def _load_config(self, config_path: str) -> Dict:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _setup_chain(
        self, api_key, base_url, device, temperature, max_tokens, load_in_4bit,
    ):
        """Set up the LLM chain based on the selected model."""
        if self.model_key not in MODELS:
            raise ValueError(
                f"Unknown model '{self.model_key}'. "
                f"Available: {list(MODELS.keys())}"
            )

        cfg = MODELS[self.model_key]
        backend = cfg["backend"]
        model_id = cfg["model_id"]

        options = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "num_ctx": 32768,
            "top_k": 40,
            "top_p": 0.9,
        }

        if backend in ("hf_causal", "hf_seq2seq"):
            # HuggingFace: use raw llm_fn + raw chain
            llm_fn = build_llm_fn(
                self.model_key, api_key=api_key, device=device,
                temperature=temperature, max_tokens=max_tokens,
                load_in_4bit=load_in_4bit,
            )
            self.chain = create_raw_chain(
                llm_fn, self.response_model, self.prompt_name
            )
        else:
            # LangChain backends (Ollama, OpenAI, Google)
            model = get_model(
                provider=backend,
                model_name=model_id,
                base_url=base_url or None,
                api_key=api_key,
                format=self.response_model.model_json_schema(),
                options=options,
            )
            self.chain = create_chain(
                model, self.response_model, self.prompt_name
            )

        logger.info(
            f"Chain ready: model={self.model_key} ({model_id}), "
            f"prompt={self.prompt_name}"
        )

    def run_timeline(
        self,
        file_path: str,
        context_window: int = 5,
    ) -> Dict[str, Any]:
        """Run assessment on every post in a single timeline file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")

        data = load_timeline_data(file_path)
        timeline_id = data.get("timeline_id", Path(file_path).stem)
        posts = data["posts"]

        logger.info(f"Timeline {timeline_id}: {len(posts)} posts")

        assessments = []
        for idx, post in enumerate(posts):
            post_id = post.get("post_id", str(idx))

            start = max(0, idx - context_window)
            window = posts[start: idx + 1]
            window_ids = [
                p.get("post_id", str(i))
                for i, p in enumerate(window, start=start)
            ]

            formatted = format_timeline_for_llm(
                window, window_ids,
                context_window=context_window,
                target_idx=len(window) - 1,
            )

            logger.info(
                f"[{timeline_id}] post {idx + 1}/{len(posts)} "
                f"(id: {post_id})..."
            )

            try:
                response = self.chain.invoke({"post_text": formatted})
                assessments.append(response.model_dump())
            except Exception as e:
                logger.error(
                    f"[{timeline_id}] post {post_id} failed: {e}. "
                    f"Using empty prediction."
                )
                # Return empty prediction to avoid crashing the whole pipeline
                assessments.append({"post_id": post_id})

        logger.info(
            f"Timeline {timeline_id} complete. "
            f"{len(assessments)} posts assessed."
        )
        return {
            "timeline_id": timeline_id,
            "num_posts": len(posts),
            "assessments": assessments,
        }

    def run_dataset(
        self,
        data_dir: str,
        context_window: int = 5,
        output_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run assessment on all timeline files in a directory."""
        files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if not files:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")

        logger.info(f"Found {len(files)} timeline files in {data_dir}")

        results = []
        for file_path in files:
            result = self.run_timeline(file_path, context_window=context_window)
            results.append(result)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results written to {output_path}")

        logger.info(f"Dataset complete. {len(results)} timelines processed.")
        return results
