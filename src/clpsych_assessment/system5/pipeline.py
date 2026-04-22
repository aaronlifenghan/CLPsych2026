"""
CLPsych 2026 — System 5 — Task 3.1 Pipeline

Generates sequence summaries using one of five prompt strategies:
  zero_shot_direct  | zero_shot_cot  | few_shot_direct
  few_shot_cot      | baseline2_style

Reuses system3's chain.py (model registry, backends, create_raw_chain).
Does NOT duplicate any model loading code.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..system3.chain import MODELS, build_llm_fn, create_chain, get_model

from .preprocessor import (
    format_sequence_for_prompt,
    load_sequences,
    load_timelines,
    resolve_sequence_posts,
)
from .few_shot_examples import build_few_shot_block, load_gold_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

WORD_LIMIT = 350

STRATEGIES = [
    "zero_shot_direct",
    "zero_shot_cot",
    "few_shot_direct",
    "few_shot_cot",
    "baseline2_style",
    "colleague_style",
]

# Strategies that use few-shot examples
FEW_SHOT_STRATEGIES = {"few_shot_direct", "few_shot_cot"}

# Strategies that need the CoT post-processing step
COT_STRATEGIES = {"zero_shot_cot", "few_shot_cot"}


class Task31Pipeline:
    """
    Pipeline for Task 3.1: sequence summarisation.

    Parameters
    ----------
    strategy : str
        One of STRATEGIES.
    model_key : str
        Key from system3.chain.MODELS (e.g. "llama3.1", "gemma2:9b").
    config_path : str, optional
        Path to system5 config.yaml.
    train_sequences_file : str, optional
        Path to train_task3.json — required for few-shot strategies.
    train_timelines_dir : str, optional
        Path to train_tasks12/ dir — required for few-shot strategies.
    api_key : str
        API key for Google/OpenAI backends.
    base_url : str
        Custom base URL for Ollama or OpenAI-compatible servers.
    device : str
        Torch device for HF models ("auto", "cuda", "cpu").
    temperature : float
    max_tokens : int
    load_in_4bit : bool
    """

    def __init__(
        self,
        strategy: str,
        model_key: str = "llama3.1",
        config_path: Optional[str] = None,
        train_sequences_file: Optional[str] = None,
        train_timelines_dir: Optional[str] = None,
        api_key: str = "",
        base_url: str = "",
        device: str = "auto",
        temperature: float = 0.2,
        max_tokens: int = 800,
        load_in_4bit: bool = False,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Available: {STRATEGIES}"
            )
        if model_key not in MODELS:
            raise ValueError(
                f"Unknown model '{model_key}'. Available: {list(MODELS.keys())}"
            )

        self.strategy = strategy
        self.model_key = model_key

        # Load config
        if config_path is None:
            # Search same candidate dirs as prompts
            for d in [
                Path(__file__).parent,
                Path.cwd() / "src" / "clpsych_assessment" / "system5",
                Path.cwd() / "system5",
            ]:
                p = d / "config.yaml"
                if p.exists():
                    config_path = str(p)
                    break
            else:
                config_path = str(Path(__file__).parent / "config.yaml")
        self.config = self._load_config(config_path)

        # Load few-shot block if needed
        self.few_shot_block = ""
        if strategy in FEW_SHOT_STRATEGIES:
            if not train_sequences_file or not train_timelines_dir:
                raise ValueError(
                    f"Strategy '{strategy}' requires --train-sequences and "
                    f"--train-timelines-dir to build few-shot examples."
                )
            few_shot_cfg = self.config.get("few_shot", {})
            gold_seqs = load_gold_sequences(train_sequences_file)
            train_timelines = load_timelines(train_timelines_dir)
            self.few_shot_block = build_few_shot_block(
                gold_sequences=gold_seqs,
                timelines=train_timelines,
                strategy=strategy,
                n=few_shot_cfg.get("n_examples", 2),
                selection=few_shot_cfg.get("selection", "diverse"),
            )
            logger.info(
                f"Few-shot block built: "
                f"{self.few_shot_block.count('### Example')} example(s)"
            )

        # Load prompt template
        # Search order:
        #   1. Relative to this file (installed package)
        #   2. Common project layouts relative to cwd
        #   3. SYSTEM5_PROMPTS_DIR env var override
        prompt_filename = f"{strategy}.md"
        candidate_dirs = [
            Path(__file__).parent / "prompts",
            Path.cwd() / "src" / "clpsych_assessment" / "system5" / "prompts",
            Path.cwd() / "system5" / "prompts",
        ]
        env_dir = os.environ.get("SYSTEM5_PROMPTS_DIR")
        if env_dir:
            candidate_dirs.insert(0, Path(env_dir))

        prompt_path = None
        for d in candidate_dirs:
            p = d / prompt_filename
            if p.exists():
                prompt_path = p
                break

        if prompt_path is None:
            searched = "\n  ".join(str(d / prompt_filename) for d in candidate_dirs)
            raise FileNotFoundError(
                f"Prompt template '{prompt_filename}' not found. Searched:\n  {searched}\n"
                f"Tip: set SYSTEM5_PROMPTS_DIR=/path/to/system5/prompts"
            )

        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        logger.info(f"Prompt loaded from: {prompt_path}")

        # Set up LLM
        self._setup_llm(api_key, base_url, device, temperature, max_tokens, load_in_4bit)

        logger.info(
            f"Task31Pipeline ready: model={model_key}, strategy={strategy}"
        )

    # ── Config ────────────────────────────────────────────────────

    def _load_config(self, config_path: str) -> Dict:
        import os
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ── LLM setup ─────────────────────────────────────────────────

    def _setup_llm(
        self, api_key, base_url, device, temperature, max_tokens, load_in_4bit
    ):
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
            self._llm_fn = build_llm_fn(
                self.model_key,
                api_key=api_key,
                device=device,
                temperature=temperature,
                max_tokens=max_tokens,
                load_in_4bit=load_in_4bit,
            )
            self._use_raw = True
        else:
            # IMPORTANT: format must be None (not "") for free-text output.
            # Passing format="" to ChatOllama activates JSON mode, which
            # causes the model to wrap its response in JSON and breaks
            # plain text extraction.
            self._model = get_model(
                provider=backend,
                model_name=model_id,
                base_url=base_url or None,
                api_key=api_key,
                format=None,
                options=options,
            )
            self._use_raw = False

    # ── Inference ─────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return raw text output."""
        if self._use_raw:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert clinical psychologist. "
                        "Follow instructions exactly."
                    ),
                },
                {"role": "user", "content": prompt},
            ]
            return self._llm_fn(messages)
        else:
            # LangChain path — use HumanMessage/SystemMessage.
            # ChatOllama.invoke() returns an AIMessage; extract .content.
            # We do NOT use a chain here (no parser, no template) — raw text only.
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(
                    content=(
                        "You are an expert clinical psychologist. "
                        "Follow instructions exactly."
                    )
                ),
                HumanMessage(content=prompt),
            ]
            response = self._model.invoke(messages)
            # AIMessage.content is a string for text models.
            # For multimodal responses it can be a list — handle both.
            if hasattr(response, "content"):
                content = response.content
                if isinstance(content, list):
                    # Extract text blocks from multimodal response
                    return " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    ).strip()
                return str(content).strip()
            return str(response).strip()

    def _build_prompt(self, sequence_text: str) -> str:
        """Fill the prompt template with sequence text and few-shot block."""
        prompt = self.prompt_template
        prompt = prompt.replace("{sequence_text}", sequence_text)
        prompt = prompt.replace("{few_shot_block}", self.few_shot_block)
        return prompt

    def _extract_summary(self, raw_output: str) -> str:
        """
        Post-process LLM output.

        For CoT strategies: extract only the text after "SUMMARY:" heading.
        For direct strategies: use the full output.
        Then enforce the 350-word limit.
        """
        text = raw_output.strip()

        if self.strategy in COT_STRATEGIES:
            # Look for SUMMARY: section (case-insensitive, with or without **)
            match = re.search(
                r"(?:^|\n)\s*\*{0,2}SUMMARY\*{0,2}:?\s*\n+(.*)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if match:
                text = match.group(1).strip()
            else:
                # SUMMARY: heading not found — model didn't follow the CoT
                # instruction. Strip the REASONING: block if present and use
                # whatever comes after it, otherwise use the full output.
                reasoning_match = re.search(
                    r"(?:^|\n)\s*\*{0,2}REASONING\*{0,2}:?\s*\n+(.*?)(?=\n\s*\*{0,2}\w+\*{0,2}:|$)",
                    text,
                    re.IGNORECASE | re.DOTALL,
                )
                if reasoning_match:
                    # Remove the reasoning block and take the rest
                    text = text[reasoning_match.end():].strip() or text
                # If still nothing useful, keep full output — never return empty

        # Enforce word limit
        words = text.split()
        if len(words) > WORD_LIMIT:
            text = " ".join(words[:WORD_LIMIT])

        return text

    # ── Public API ────────────────────────────────────────────────

    def summarise_sequence(
        self,
        posts: List[Dict[str, Any]],
        timeline_id: str,
        sequence_id: str,
    ) -> str:
        """Generate a summary for one sequence."""
        sequence_text = format_sequence_for_prompt(posts, self.strategy)
        prompt = self._build_prompt(sequence_text)

        try:
            raw = self._call_llm(prompt)
            summary = self._extract_summary(raw)
            return summary
        except Exception as e:
            logger.error(
                f"[{timeline_id}/{sequence_id}] LLM call failed: {e}. "
                f"Returning empty summary."
            )
            return ""

    def run(
        self,
        sequences_file: str,
        timelines_dir: str,
        output_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Run Task 3.1 over all sequences in sequences_file.

        Parameters
        ----------
        sequences_file : str
            test_task3nolabels.json or train_task3.json (for train-split eval)
        timelines_dir : str
            Directory with per-timeline JSONs (train_tasks12 or test_tasks12nolabels)
        output_path : str
            Where to write raw_task3.json

        Returns
        -------
        List of {timeline_id, sequence_id, summary} dicts
        """
        sequences = load_sequences(sequences_file)
        timelines = load_timelines(timelines_dir)

        logger.info(
            f"Running Task 3.1: {len(sequences)} sequences, "
            f"model={self.model_key}, strategy={self.strategy}"
        )

        results = []
        for i, seq in enumerate(sequences, 1):
            tid = seq["timeline_id"]
            seq_id = seq["sequence_id"]
            posts = resolve_sequence_posts(seq, timelines)

            if not posts:
                logger.warning(f"  [{i}/{len(sequences)}] {tid}/{seq_id}: no posts resolved, skipping")
                continue

            logger.info(
                f"  [{i}/{len(sequences)}] {tid}/{seq_id} "
                f"({len(posts)} posts)..."
            )

            summary = self.summarise_sequence(posts, tid, seq_id)
            results.append({
                "timeline_id": tid,
                "sequence_id": seq_id,
                "summary": summary,
            })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Done. {len(results)} summaries written to {output_path}"
        )
        return results