import os
import time
import logging
import traceback
from typing import Dict, Any, Optional, Tuple

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
        "provider": "novita",
        "task": "conversational",
        # Цены за МИЛЛИОН токенов (в $, взято с https://novita.ai/pricing)
        "input_price_per_mt": 0.28,   # $0.28 за 1 млн входных токенов
        "output_price_per_mt": 0.42   # $0.42 за 1 млн выходных токенов
    },
    "orpo": {
        "id": "AdamLucek/Orpo-Llama-3.2-1B-15k",
        "provider": "featherless-ai",
        "task": "conversational",
        "is_free": True  # Бесплатная модель
    },
    "qwen2_5": {
        "id": "adrimoreau/Qwen2.5-0.5B-Instruct-Gensyn-Swarm-untamed_insectivorous_coyote",
        "provider": "featherless-ai",
        "task": "conversational",
        "is_free": True  # Бесплатная модель
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
    """Приблизительный подсчет токенов"""
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except:
            pass
    # Fallback: приблизительная оценка
    return max(1, int(len(text.split()) * 1.3))


def estimate_cost(input_tokens: int, output_tokens: int, model_info: Dict) -> Optional[float]:
    """Оценка стоимости на основе входных и выходных токенов"""

    # Если модель бесплатная
    if model_info.get("is_free", False):
        return 0.0

    # Если указаны цены за миллион токенов (для платных моделей)
    input_price = model_info.get("input_price_per_mt")
    output_price = model_info.get("output_price_per_mt")

    if input_price is None or output_price is None:
        logger.warning(f"No pricing info for model {model_info.get('id')}")
        return None

    # Расчет: (токены / 1,000,000) * цена
    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    logger.info(f"Cost calculation: {input_tokens} input tokens -> ${input_cost:.6f}, "
                f"{output_tokens} output tokens -> ${output_cost:.6f}, total: ${total_cost:.6f}")

    return total_cost


def get_model_display_info(model_key: str, model_info: Dict) -> Dict:
    """Получить информацию о модели для отображения в интерфейсе"""
    info = {
        "id": model_info["id"],
        "provider": model_info["provider"],
        "task": model_info.get("task", "text-generation"),
        "is_free": model_info.get("is_free", False)
    }

    if model_key == "deepseek":
        info["pricing"] = {
            "input": "$0.28 per 1M tokens",
            "output": "$0.42 per 1M tokens"
        }
    elif model_info.get("is_free", False):
        info["pricing"] = "Free (with limitations)"

    return info


# -----------------------------
# FLASK
# -----------------------------
app = Flask(__name__)


@app.route('/')
def index():
    logger.info("GET / — sending index.html")
    return send_from_directory('public', 'index.html')


@app.route('/api/models', methods=['GET'])
def models_list():
    logger.info("GET /api/models")
    models_info = {}
    for key, info in MODELS.items():
        models_info[key] = get_model_display_info(key, info)
    return jsonify(models_info)


@app.route('/api/generate', methods=['POST'])
def generate():
    logger.info("POST /api/generate")

    body = request.get_json() or {}
    logger.info("Incoming request for model: %s", body.get("model_key"))

    model_key = body.get("model_key")
    prompt = (body.get("prompt") or "").strip()
    max_new_tokens = int(body.get("max_tokens", 200))

    if not model_key or model_key not in MODELS:
        logger.warning("Unknown model_key: %s", model_key)
        return jsonify({"error": "Unknown model_key"}), 400

    if not prompt:
        logger.warning("Empty prompt")
        return jsonify({"error": "Empty prompt"}), 400

    model_info = MODELS[model_key]
    model_id = model_info["id"]
    task_type = model_info.get("task", "text-generation")

    logger.info("Selected model: %s (%s), task: %s", model_key, model_id, task_type)
    logger.info("Prompt length: %d chars", len(prompt))

    # Оценка входных токенов
    input_tokens = approx_token_count(prompt, model_id=model_id)
    logger.info("Estimated input tokens: %d", input_tokens)

    start_time = time.time()
    elapsed = 0
    response_text = ""

    try:
        logger.info("Sending request to HF model: %s (task: %s)", model_id, task_type)

        if task_type == "conversational":
            # Для conversational моделей используем другой формат запроса
            out = client.conversational(
                model=model_id,
                inputs={
                    "text": prompt,
                    "parameters": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.7,
                        "top_p": 0.95
                    }
                }
            )

            # Парсим ответ от conversational API
            if isinstance(out, dict) and "generated_text" in out:
                response_text = out["generated_text"]
            elif isinstance(out, dict) and "conversation" in out:
                if "generated_responses" in out["conversation"] and out["conversation"]["generated_responses"]:
                    response_text = out["conversation"]["generated_responses"][0]
            else:
                response_text = str(out)

        elif task_type == "text-generation":
            # Для text-generation моделей
            out = client.text_generation(
                model=model_id,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.95,
                stream=False
            )

            # Парсим ответ от text-generation API
            if hasattr(out, 'generated_text'):
                response_text = out.generated_text
            elif isinstance(out, dict) and "generated_text" in out:
                response_text = out["generated_text"]
            elif isinstance(out, str):
                response_text = out
            else:
                response_text = str(out)

        elapsed = time.time() - start_time
        logger.info("HF responded in %.3f sec", elapsed)

        # Логируем ответ (обрезаем для читаемости)
        if response_text and len(response_text) > 500:
            logger.info("Response (first 500 chars): %s...", response_text[:500])
        elif response_text:
            logger.info("Response: %s", response_text)
        else:
            logger.warning("Empty response received")

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error("HF inference error after %.3f sec: %s", elapsed, str(e))
        logger.error(traceback.format_exc())

        # Попробуем альтернативный метод для conversational моделей
        if task_type == "conversational":
            try:
                logger.info("Trying alternative method for conversational model...")
                alt_start = time.time()

                # Альтернативный способ через chat completion
                messages = [{"role": "user", "content": prompt}]
                out = client.chat_completion(
                    model=model_id,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0.7
                )

                if isinstance(out, dict) and "choices" in out:
                    response_text = out["choices"][0]["message"]["content"]
                elapsed = time.time() - alt_start
                logger.info("Alternative method succeeded in %.3f sec", elapsed)

            except Exception as alt_e:
                logger.error("Alternative method also failed: %s", str(alt_e))
                return jsonify({
                    "error": f"Model {model_id} requires conversational API. Try another model.",
                    "details": str(e)
                }), 500

        else:
            return jsonify({
                "error": "HF inference error",
                "details": str(e)
            }), 500

    # Удаляем дублирование промпта в ответе (если есть)
    if response_text and response_text.startswith(prompt):
        response_text = response_text[len(prompt):].strip()

    output_tokens = approx_token_count(response_text, model_id=model_id)
    total_tokens = input_tokens + output_tokens

    logger.info("Output tokens: %d, total: %d", output_tokens, total_tokens)

    cost = estimate_cost(input_tokens, output_tokens, model_info)

    result = {
        "model_key": model_key,
        "model_id": model_id,
        "model_provider": model_info["provider"],
        "is_free": model_info.get("is_free", False),
        "prompt": prompt,
        "response_text": response_text,
        "time_s": round(elapsed, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 8) if cost is not None else None,
        "task_type": task_type,
        "pricing_info": get_model_display_info(model_key, model_info).get("pricing", "Unknown")
    }

    logger.info("Returning result with time: %.3fs, tokens: %d, cost: $%.8f",
                elapsed, total_tokens, cost or 0.0)
    return jsonify(result)


@app.route('/api/pricing_info/<model_key>', methods=['GET'])
def pricing_info(model_key):
    """Получить детальную информацию о ценообразовании для модели"""
    if model_key not in MODELS:
        return jsonify({"error": "Model not found"}), 404

    model_info = MODELS[model_key]
    info = get_model_display_info(model_key, model_info)

    # Добавляем детали расчета для DeepSeek
    if model_key == "deepseek":
        info["calculation_example"] = {
            "example_input_tokens": 1000,
            "example_output_tokens": 1000,
            "input_cost": f"${(1000/1_000_000 * 0.28):.6f}",
            "output_cost": f"${(1000/1_000_000 * 0.42):.6f}",
            "total_cost": f"${(1000/1_000_000 * 0.28 + 1000/1_000_000 * 0.42):.6f}"
        }

    return jsonify(info)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=True)