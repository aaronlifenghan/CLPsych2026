import glob
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import yaml
from pydantic import BaseModel

from .chain import create_chain, get_model
from .preprocessor import (
    format_timeline_for_llm,
    load_timeline_data,
)

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
        config_path: Optional[str] = None,
    ):
        if config_path is None:
            config_path = str(Path(__file__).parent / "config.yaml")

        self.config = self._load_config(config_path)
        self.provider = self.config["default_provider"]
        self.response_model = response_model
        self.prompt_name = prompt_name
        self._setup_chain()

    def _load_config(self, config_path: str) -> Dict:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    def _setup_chain(self):
        provider_cfg = self.config["providers"][self.provider]
        model = get_model(
            self.provider,
            base_url=provider_cfg.get("base_url"),
            model_name=provider_cfg["model_name"],
            format=self.response_model.model_json_schema(),
            options=provider_cfg.get("options", {}),
        )
        self.chain = create_chain(
            model, self.response_model, self.prompt_name
        )
        logger.info(
            f"Chain ready: prompt={self.prompt_name}, "
            f"model={provider_cfg['model_name']}"
        )

    def run_timeline(
        self,
        file_path: str,
        context_window: int = 5,
    ) -> Dict[str, Any]:
        """
        Run assessment on every post in a single timeline file.

        Args:
            file_path: Path to a JSON timeline file.
            context_window: Number of preceding posts for context.

        Returns:
            Dict with timeline_id, num_posts, and assessments.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Input file not found: {file_path}"
            )

        data = load_timeline_data(file_path)
        timeline_id = data.get("timeline_id", Path(file_path).stem)
        posts = data["posts"]

        logger.info(
            f"Timeline {timeline_id}: {len(posts)} posts"
        )

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
                window,
                window_ids,
                context_window=context_window,
                target_idx=len(window) - 1,
            )

            logger.info(
                f"[{timeline_id}] post {idx + 1}/{len(posts)} "
                f"(id: {post_id})..."
            )

            response = self.chain.invoke({"post_text": formatted})
            assessments.append(response.model_dump())

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
        """
        Run assessment on all timeline files in a directory.

        Args:
            data_dir: Path to directory containing JSON timeline files.
            context_window: Number of preceding posts for context.
            output_path: If provided, write results to this JSON file.

        Returns:
            List of timeline result dicts.
        """
        files = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        if not files:
            raise FileNotFoundError(
                f"No JSON files found in {data_dir}"
            )

        logger.info(f"Found {len(files)} timeline files in {data_dir}")

        results = []
        for file_path in files:
            result = self.run_timeline(
                file_path, context_window=context_window
            )
            results.append(result)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results written to {output_path}")

        logger.info(
            f"Dataset complete. {len(results)} timelines processed."
        )
        return results
