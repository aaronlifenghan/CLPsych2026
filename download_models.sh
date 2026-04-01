#!/usr/bin/env bash
# download_models.sh — Pull all Ollama models and download HuggingFace models
#
# Usage:
#   ./download_models.sh              # download all
#   ./download_models.sh ollama       # only Ollama models
#   ./download_models.sh hf           # only HuggingFace models
#   ./download_models.sh hf 2         # HF models with 2 parallel downloads

set -euo pipefail

TARGET="${1:-all}"
PARALLEL="${2:-1}"

# ── Ollama models ────────────────────────────────────────────────

OLLAMA_MODELS=(
    "llama3.1:8b"
    "gemma2:9b"
    "gemma2:27b"
    "qwen2.5:7b"
    "qwen2.5:32b"
    "phi3:14b"
    "mistral:7b"
    "mistral-nemo"
)

download_ollama() {
    echo ""
    echo "=== Ollama models ==="

    if ! command -v ollama &>/dev/null; then
        echo "ERROR: ollama not found. Install from https://ollama.com"
        return 1
    fi

    for model in "${OLLAMA_MODELS[@]}"; do
        echo "[Ollama] Pulling $model ..."
        ollama pull "$model" || echo "  WARN: Failed to pull $model"
    done

    echo "Ollama models done."
}

# ── HuggingFace models ───────────────────────────────────────────

HF_MODELS=(
    "klyang/MentalLLaMA-chat-13B"
    "AIMH/SQPsychLLM-8b-gemma"
    "NEU-HAI/mental-alpaca"
    "NEU-HAI/mental-flan-t5-large"
)

download_one_hf_model() {
    local repo="$1"
    local name
    name=$(basename "$repo")
    echo "[HF] Downloading $repo ..."
    huggingface-cli download "$repo" \
        --local-dir "models/hf_cache/$name" \
        --local-dir-use-symlinks False \
        2>&1 | tail -3
    echo "[HF] Done: $repo"
}

export -f download_one_hf_model

download_hf() {
    echo ""
    echo "=== HuggingFace models (parallel=$PARALLEL) ==="

    if ! command -v huggingface-cli &>/dev/null; then
        echo "ERROR: huggingface-cli not found. Run: pip install huggingface_hub"
        return 1
    fi

    printf '%s\n' "${HF_MODELS[@]}" | \
        xargs -P"$PARALLEL" -I{} bash -c 'download_one_hf_model "$@"' _ {}

    echo "HuggingFace models done."
    echo "Models cached in: models/hf_cache/"
}

# ── Python dependencies ──────────────────────────────────────────

install_deps() {
    echo ""
    echo "=== Installing Python dependencies ==="
    pip install --quiet langchain langchain-ollama langchain-openai \
        langchain-google-genai pydantic pyyaml numpy scipy scikit-learn
    echo "Dependencies installed."
}

# ── Run ──────────────────────────────────────────────────────────

mkdir -p models/hf_cache

case "$TARGET" in
    all)
        install_deps
        download_ollama || true
        download_hf
        ;;
    ollama)
        download_ollama
        ;;
    hf)
        download_hf
        ;;
    deps)
        install_deps
        ;;
    *)
        echo "Usage: $0 [all|ollama|hf|deps] [parallel_hf_jobs]"
        exit 1
        ;;
esac

echo ""
echo "All downloads complete."
echo ""
echo "Next steps:"
echo "  Ollama models are ready. Ensure 'ollama serve' is running."
echo "  For Gemini: export GOOGLE_API_KEY=your-key-here"
echo "  For OpenAI: export OPENAI_API_KEY=your-key-here"
echo "  For HF models, use: --model mentalllama --device cuda"
