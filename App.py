import os
import time
import logging
import traceback
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# -----------------------------
# ЛОГИРОВАНИЕ — ВКЛЮЧАЕМ
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------
# ЗАГРУЗКА .env
# -----------------------------
try:
    load_dotenv()
    logger.info(".env loaded successfully")
except Exception as e:
    logger.error(f"Failed to load .env: {e}")

HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN")
if not HF_TOKEN:
    logger.error("HUGGINGFACE_API_TOKEN is missing!")
    raise RuntimeError("Set HUGGINGFACE_API_TOKEN in environment or .env")

logger.info("HF token loaded (first 6 chars): %s******", HF_TOKEN[:6])

# -----------------------------
# CONFIG МОДЕЛЕЙ
# -----------------------------
MODELS = {
    "deepseek": {
        "id": "deepseek-ai/DeepSeek-V3.2",
        "price_per_1k": None,
        "provider": "novita"
    },
    "orpo": {
        "id": "AdamLucek/Orpo-Llama-3.2-1B-15k",
        "price_per_1k": None,
        "provider": "featherless-ai"
    },
    "qwen2_5": {
        "id": "adrimoreau/Qwen2.5-0.5B-Instruct-Gensyn-Swarm-untamed_insectivorous_coyote",
        "price_per_1k": None,
        "provider": "featherless-ai"
    }
}

logger.info("Loaded %d models: %s", len(MODELS), list(MODELS.keys()))

# -----------------------------
# ИНИЦИАЛИЗАЦИЯ КЛИЕНТА HF
# -----------------------------
client = InferenceClient(token=HF_TOKEN)
logger.info("InferenceClient initialized")

# -----------------------------
# TOKENS ESTIMATION
# -----------------------------
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    logger.info("tiktoken enabled")
except Exception:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken unavailable — using simple token estimator")


def approx_token_count(text: str, model_id: Optional[str] = None) -> int:
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except:
            pass
    return max(1, int(len(text.split()) * 1.3))


def estimate_cost(tokens: int, price_per_1k: Optional[float]) -> Optional[float]:
    if price_per_1k is None:
        return None
    return (tokens / 1000) * price_per_1k


# -----------------------------
# FLASK
# -----------------------------
app = Flask(__name__, static_folder='public', static_url_path='')


@app.route('/')
def index():
    logger.info("GET / — sending index.html")
    return send_from_directory('public', 'index.html')


@app.route('/api/models', methods=['GET'])
def models_list():
    logger.info("GET /api/models")
    return jsonify({k: {"id": v["id"], "price_per_1k": v["price_per_1k"]} for k, v in MODELS.items()})


@app.route('/api/generate', methods=['POST'])
def generate():
    logger.info("POST /api/generate")

    body = request.get_json() or {}
    logger.info("Incoming JSON: %s", body)

    model_key = body.get("model_key")
    prompt = (body.get("prompt") or "").strip()
    max_new_tokens = int(body.get("max_tokens", 512))

    if not model_key or model_key not in MODELS:
        logger.warning("Unknown model_key: %s", model_key)
        return jsonify({"error": "Unknown model_key"}), 400

    if not prompt:
        logger.warning("Empty prompt")
        return jsonify({"error": "Empty prompt"}), 400

    model_info = MODELS[model_key]
    model_id = model_info["id"]

    logger.info("Selected model: %s (%s)", model_key, model_id)

    input_tokens = approx_token_count(prompt, model_id=model_id)
    logger.info("Estimated input tokens: %d", input_tokens)

    start = time.perf_counter()

    try:
        logger.info("Sending request to HF model: %s", model_id)
        out = client.text_generation(
            model=model_id,
            inputs=prompt,
            max_new_tokens=max_new_tokens
        )
        elapsed = time.perf_counter() - start
        logger.info("HF responded in %.3f sec", elapsed)
        logger.info("Raw HF response: %s", str(out)[:500])

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.error("HF inference error after %.3f sec: %s", elapsed, str(e))
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "HF inference error",
            "details": str(e),
            "trace": traceback.format_exc()
        }), 500

    # -----------------------------
    # PARSE RESPONSE
    # -----------------------------
    if isinstance(out, dict) and "generated_text" in out:
        response_text = out["generated_text"]
    elif isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
        response_text = out[0]["generated_text"]
    elif isinstance(out, str):
        response_text = out
    else:
        response_text = str(out)

    output_tokens = approx_token_count(response_text, model_id=model_id)
    total_tokens = input_tokens + output_tokens

    logger.info("Output tokens: %d, total: %d", output_tokens, total_tokens)

    cost = estimate_cost(total_tokens, model_info.get("price_per_1k"))

    result = {
        "model_key": model_key,
        "model_id": model_id,
        "prompt": prompt,
        "response_text": response_text,
        "time_s": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": cost
    }

    logger.info("Returning result to frontend: %s", str(result)[:400])
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port)