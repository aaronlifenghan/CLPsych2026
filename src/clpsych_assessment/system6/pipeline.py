"""
CLPsych 2026 — System 6 — RAG-enhanced Task 3.1 Pipeline

Key difference from System 5:
  System 5 uses fixed, static few-shot examples (always the same 2 examples).
  System 6 dynamically retrieves the most semantically similar examples
  from the training library for each query sequence.

Prompt strategies (all RAG-enhanced):
  rag_few_shot_direct   — adapted from system5's best llama prompt
  rag_baseline2_style   — adapted from system5's best gemma4 prompt (enriched ABCD)
  rag_colleague_style   — adapted from system5's colleague prompt

Reuses system3's chain.py for model backends.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..system3.chain import MODELS, build_llm_fn, create_chain, get_model

from ..system5.preprocessor import (
    format_sequence_for_prompt,
    load_sequences,
    load_timelines,
    resolve_sequence_posts,
)

from .rag_index import RAGIndex, build_rag_index
from .rag_retriever import RAGRetriever, format_retrieved_examples

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

WORD_LIMIT = 350

STRATEGIES = [
    "rag_few_shot_direct",
    "rag_baseline2_style",
    "rag_colleague_style",
]


class RAGTask31Pipeline:
    """
    RAG-enhanced pipeline for Task 3.1: sequence summarisation.

    Parameters
    ----------
    strategy : str
        One of STRATEGIES.
    model_key : str
        Key from system3.chain.MODELS.
    config_path : str, optional
        Path to system6 config.yaml.
    train_sequences_file : str
        Path to train_task3_train_fold.json (library source).
    train_timelines_dir : str
        Path to train_tasks12/ directory.
    api_key, base_url, device, temperature, max_tokens, load_in_4bit :
        Model configuration (passed to system3 chain).
    rag_top_k : int, optional
        Override config's rag.top_k.
    rag_strategy : str, optional
        Override config's rag.retrieval_strategy.
    embedding_backend : str, optional
        Override config's rag.embedding_backend.
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
        rag_top_k: Optional[int] = None,
        rag_strategy: Optional[str] = None,
        embedding_backend: Optional[str] = None,
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
            for d in [
                Path(__file__).parent,
                Path.cwd() / "src" / "clpsych_assessment" / "system6",
                Path.cwd() / "system6",
            ]:
                p = d / "config.yaml"
                if p.exists():
                    config_path = str(p)
                    break
            else:
                config_path = str(Path(__file__).parent / "config.yaml")
        self.config = self._load_config(config_path)
        rag_cfg = self.config.get("rag", {})

        # Load prompt template
        prompt_filename = f"{strategy}.md"
        candidate_dirs = [
            Path(__file__).parent / "prompts",
            Path.cwd() / "src" / "clpsych_assessment" / "system6" / "prompts",
            Path.cwd() / "system6" / "prompts",
        ]
        env_dir = os.environ.get("SYSTEM6_PROMPTS_DIR")
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
                f"Prompt template '{prompt_filename}' not found. Searched:\n  {searched}"
            )
        self.prompt_template = prompt_path.read_text(encoding="utf-8")
        logger.info(f"Prompt loaded from: {prompt_path}")

        # Build or load RAG index
        if not train_sequences_file or not train_timelines_dir:
            raise ValueError(
                "RAG strategies require --train-sequences and --train-timelines-dir"
            )

        actual_top_k = rag_top_k or rag_cfg.get("top_k", 3)
        actual_rag_strategy = rag_strategy or rag_cfg.get("retrieval_strategy", "hybrid")
        actual_embedding = embedding_backend or rag_cfg.get("embedding_backend", "auto")

        index_cache = rag_cfg.get("index_cache_path")
        if index_cache and Path(index_cache).exists():
            logger.info(f"Loading cached RAG index from {index_cache}")
            try:
                self.rag_index = RAGIndex.load(index_cache)
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}. Rebuilding...")
                self.rag_index = self._build_index(
                    train_sequences_file, train_timelines_dir,
                    actual_embedding, rag_cfg, index_cache,
                )
        else:
            self.rag_index = self._build_index(
                train_sequences_file, train_timelines_dir,
                actual_embedding, rag_cfg, index_cache,
            )

        # Build retriever
        self.retriever = RAGRetriever(
            index=self.rag_index,
            top_k=actual_top_k,
            strategy=actual_rag_strategy,
            change_type_boost=rag_cfg.get("change_type_boost", 0.15),
            sequence_only=rag_cfg.get("sequence_only", True),
            diversity_penalty=rag_cfg.get("diversity_penalty", 0.3),
            use_clinical_text=rag_cfg.get("use_clinical_text", True),
        )
        self.include_posts_in_examples = rag_cfg.get("include_posts_in_examples", True)

        logger.info(
            f"RAG retriever ready: top_k={actual_top_k}, "
            f"strategy={actual_rag_strategy}, "
            f"index_size={len(self.rag_index.entries)}"
        )

        # Set up LLM
        self._setup_llm(api_key, base_url, device, temperature, max_tokens, load_in_4bit)

        logger.info(
            f"RAGTask31Pipeline ready: model={model_key}, strategy={strategy}"
        )

    # ── Config ────────────────────────────────────────────────────

    def _load_config(self, config_path: str) -> Dict:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def _build_index(
        self,
        train_sequences_file: str,
        train_timelines_dir: str,
        embedding_backend: str,
        rag_cfg: Dict,
        cache_path: Optional[str],
    ) -> RAGIndex:
        """Build RAG index from training data."""
        return build_rag_index(
            train_sequences_file=train_sequences_file,
            timelines_dir=train_timelines_dir,
            embedding_backend=embedding_backend,
            include_post_groups=rag_cfg.get("include_post_groups", False),
            use_clinical_text=rag_cfg.get("use_clinical_text", True),
            save_path=cache_path,
        )

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
            if hasattr(response, "content"):
                content = response.content
                if isinstance(content, list):
                    return " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    ).strip()
                return str(content).strip()
            return str(response).strip()

    def _build_prompt(
        self,
        sequence_text: str,
        rag_examples_block: str,
    ) -> str:
        """Fill the prompt template with sequence text and RAG examples."""
        prompt = self.prompt_template
        prompt = prompt.replace("{sequence_text}", sequence_text)
        prompt = prompt.replace("{rag_examples_block}", rag_examples_block)
        return prompt

    def _extract_summary(self, raw_output: str) -> str:
        """Post-process LLM output and enforce 350-word limit."""
        text = raw_output.strip()

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
        change_type: Optional[str] = None,
    ) -> str:
        """Generate a summary for one sequence using RAG."""
        # Step 1: Retrieve similar examples
        retrieved = self.retriever.retrieve(
            posts=posts,
            query_change_type=change_type,
            query_timeline_id=timeline_id,
        )

        # Step 2: Format retrieved examples
        rag_block = format_retrieved_examples(
            retrieved,
            strategy=self.strategy,
            include_posts=self.include_posts_in_examples,
        )

        # Step 3: Format query sequence
        # Determine base strategy for formatting
        if "baseline2" in self.strategy:
            base_strategy = "baseline2_style"
        else:
            base_strategy = "zero_shot_direct"  # default formatting
        sequence_text = format_sequence_for_prompt(posts, base_strategy)

        # Step 4: Build and call prompt
        prompt = self._build_prompt(sequence_text, rag_block)

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
            test_task3nolabels.json or train_task3_val_fold.json
        timelines_dir : str
            Directory with per-timeline JSONs
        output_path : str
            Where to write raw_task3.json

        Returns
        -------
        List of {timeline_id, sequence_id, summary} dicts
        """
        sequences = load_sequences(sequences_file)
        timelines = load_timelines(timelines_dir)

        logger.info(
            f"Running RAG Task 3.1: {len(sequences)} sequences, "
            f"model={self.model_key}, strategy={self.strategy}"
        )

        results = []
        for i, seq in enumerate(sequences, 1):
            tid = seq["timeline_id"]
            seq_id = seq["sequence_id"]
            change_type = seq.get("change_type")  # may be None for test set
            posts = resolve_sequence_posts(seq, timelines)

            if not posts:
                logger.warning(
                    f"  [{i}/{len(sequences)}] {tid}/{seq_id}: no posts resolved, skipping"
                )
                continue

            logger.info(
                f"  [{i}/{len(sequences)}] {tid}/{seq_id} "
                f"({len(posts)} posts, change_type={change_type or '?'})..."
            )

            summary = self.summarise_sequence(posts, tid, seq_id, change_type)
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
