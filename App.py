import logging
import os
import time
import traceback
from typing import Dict, Optional, Any

from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory
from huggingface_hub import InferenceClient

# -----------------------------
# ЛОГИРОВАНИЕ
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
# CONFIG МОДЕЛЕЙ (ОБНОВЛЕННЫЙ ПОРЯДОК)
# -----------------------------
MODELS = {
    "deepseek": {
        "id": "deepseek-ai/DeepSeek-V3.2:novita",
        "provider": "novita",
        "task": "chat_completion",
        "api_method": "chat_completions_create",
        "input_price_per_mt": 0.28,
        "output_price_per_mt": 0.42,
        "max_tokens": 2000,
        "requires_formatting": False,
        "model_size": "685B params",
        "description": "DeepSeek V3.2 - мощная модель через Novita"
    },
    "smollm3": {
        "id": "HuggingFaceTB/SmolLM3-3B:hf-inference",
        "provider": "hf-inference",
        "task": "chat_completion",
        "api_method": "chat_completions_create",
        "is_free": True,
        "requires_formatting": False,
        "max_tokens": 2000,
        "model_size": "3B params",
        "description": "SmolLM3-3B - эффективная 3B параметровая модель от Hugging Face",
        "context_window": 8192
    },
    "qwen2_5": {
        "id": "adrimoreau/Qwen2.5-0.5B-Instruct-Gensyn-Swarm-untamed_insectivorous_coyote:featherless-ai",
        "provider": "featherless-ai",
        "task": "chat_completion",
        "api_method": "chat_completions_create",
        "is_free": False,
        "max_tokens": 2000,
        "input_price_per_mt": 0.01,
        "output_price_per_mt": 0.015,
        "model_size": "0.5B params",
        "description": "Qwen 2.5 0.5B Instruct через Featherless AI"
    }
}

SYSTEM_PROMPT = """Вы полезный AI-ассистент. Вы предоставляете четкие, точные и лаконичные ответы.
Всегда отвечайте на том же языке, на котором задан вопрос.
Будьте дружелюбны и профессиональны в своих ответах."""

# Определяем порядок моделей для отображения
MODEL_ORDER = ["deepseek", "smollm3", "qwen2_5"]

logger.info("Loaded %d models: %s", len(MODELS), list(MODELS.keys()))

# -----------------------------
# ИНИЦИАЛИЗАЦИЯ КЛИЕНТОВ
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


def approx_token_count(text: str) -> int:
    """Приблизительный подсчет токенов"""
    if not text:
        return 0
    if TIKTOKEN_AVAILABLE:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except:
            pass
    return max(1, int(len(text.split()) * 1.3))


def extract_response_from_completion(completion: Any) -> str:
    """Универсальный метод извлечения текста ответа"""
    try:
        # Если это dict
        if isinstance(completion, dict):
            if "choices" in completion and completion["choices"]:
                choice = completion["choices"][0]
                if isinstance(choice, dict):
                    if "message" in choice:
                        message = choice["message"]
                        if isinstance(message, dict) and "content" in message:
                            return str(message["content"])
                        else:
                            return str(message)
                    elif "text" in choice:
                        return str(choice["text"])
                    elif "content" in choice:
                        return str(choice["content"])

            for key in ["generated_text", "text", "response", "output"]:
                if key in completion:
                    return str(completion[key])

        # Если это объект с атрибутами
        elif hasattr(completion, 'choices') and completion.choices:
            choice = completion.choices[0]

            # Логируем для отладки
            logger.info(f"Choice type: {type(choice)}")

            if hasattr(choice, 'message'):
                message = choice.message
                logger.info(f"Message type: {type(message)}")

                if hasattr(message, 'content'):
                    content = message.content
                    logger.info(f"Content type: {type(content)}")
                    if content is not None:
                        return str(content)
                    else:
                        # Если content None, пробуем получить message как строку
                        return str(message)
                else:
                    # Если нет content, возвращаем сам message как строку
                    return str(message)

            elif hasattr(choice, 'text'):
                return str(choice.text)

            # Попробуем напрямую получить текст
            elif hasattr(choice, '__str__'):
                return str(choice)

        # Если это строка
        elif isinstance(completion, str):
            return completion

        # Последняя попытка - преобразовать в строку
        logger.info(f"Converting completion to string: {str(completion)[:200]}")
        return str(completion)

    except Exception as e:
        logger.error(f"Error extracting response: {e}")
        return ""


def extract_smollm3_response(completion: Any) -> str:
    """Специальный метод извлечения ответа для SmolLM3"""
    try:
        # Пробуем стандартный метод
        response = extract_response_from_completion(completion)

        # Если получили непустой ответ, возвращаем его
        if response and len(response.strip()) >= 2:
            return response

        logger.warning("Standard extraction returned empty/short response")

        # Метод 1: как в примере HuggingFace
        if hasattr(completion, 'choices') and completion.choices:
            message = completion.choices[0].message
            if hasattr(message, 'content') and message.content:
                return str(message.content)
            elif hasattr(message, '__str__'):
                return str(message)

        # Метод 2: если это объект с методом dict()
        if hasattr(completion, 'dict'):
            completion_dict = completion.dict()
            if 'choices' in completion_dict and completion_dict['choices']:
                choice = completion_dict['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    return str(choice['message']['content'])

        # Последняя попытка - вернуть строковое представление
        return str(completion)

    except Exception as e:
        logger.error(f"Error extracting SmolLM3 response: {e}")
        logger.error(traceback.format_exc())
        return ""


def cleanup_response(response: str, prompt: str, model_info: Dict) -> str:
    """Очистка ответа от артефактов форматирования"""
    if not response:
        return ""

    response = response.strip()

    # Удаляем возможные теги форматирования
    cleanup_patterns = [
        "<|im_start|>", "<|im_end|>",
        "<|user|>", "<|assistant|>",
        "### Instruction:", "### Response:",
    ]

    for pattern in cleanup_patterns:
        response = response.replace(pattern, "")

    # Заменяем три переноса на два
    response = response.replace("\n\n\n", "\n\n")

    return response.strip()

def estimate_cost(input_tokens: int, output_tokens: int, model_info: Dict) -> Optional[float]:
    """Оценка стоимости"""
    if model_info.get("is_free", False):
        return 0.0

    input_price = model_info.get("input_price_per_mt")
    output_price = model_info.get("output_price_per_mt")

    if input_price is None or output_price is None:
        logger.warning(f"No pricing info for model {model_info.get('id')}")
        return None

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price
    total_cost = input_cost + output_cost

    logger.info(f"Cost calculation: {input_tokens} input tokens -> ${input_cost:.6f}, "
                f"{output_tokens} output tokens -> ${output_cost:.6f}, total: ${total_cost:.6f}")

    return total_cost


def get_model_display_info(model_key: str, model_info: Dict) -> Dict:
    """Информация о модели для интерфейса (ДОБАВЛЕН model_size)"""
    info = {
        "id": model_info["id"],
        "provider": model_info["provider"],
        "task": model_info["task"],
        "api_method": model_info.get("api_method", "chat_completion"),
        "is_free": model_info.get("is_free", False),
        "max_tokens": model_info.get("max_tokens", 200),
        "requires_formatting": model_info.get("requires_formatting", False),
        "description": model_info.get("description", ""),
        "model_size": model_info.get("model_size", "Unknown")
    }

    if "context_window" in model_info:
        info["context_window"] = model_info["context_window"]

    if model_info.get("is_free", False):
        info["pricing_info"] = f"Free (max {info['max_tokens']} tokens)"
    elif model_info.get("input_price_per_mt"):
        input_price = model_info["input_price_per_mt"]
        output_price = model_info.get("output_price_per_mt", input_price)
        info["pricing_info"] = f"Input: ${input_price}/1M, Output: ${output_price}/1M"
    else:
        info["pricing_info"] = "Paid (check provider)"

    return info


# -----------------------------
# FLASK
# -----------------------------
app = Flask(__name__)


@app.route('/')
def index():
    return send_from_directory('public', 'index.html')


@app.route('/api/models', methods=['GET'])
def models_list():
    """Возвращает модели в заданном порядке"""
    models_info = {}
    # Используем определенный порядок MODELS_ORDER
    for key in MODEL_ORDER:
        if key in MODELS:
            info = MODELS[key]
            models_info[key] = get_model_display_info(key, info)
    return jsonify(models_info)


@app.route('/api/generate', methods=['POST'])
def generate():
    logger.info("POST /api/generate")

    body = request.get_json() or {}
    logger.info("Incoming request for model: %s", body.get("model_key"))

    model_key = body.get("model_key")
    prompt = (body.get("prompt") or "").strip()
    max_tokens = int(body.get("max_tokens", 1200))

    if not model_key or model_key not in MODELS:
        return jsonify({"error": "Unknown model_key"}), 400

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    model_info = MODELS[model_key]
    model_id = model_info["id"]
    task_type = model_info["task"]
    api_method = model_info.get("api_method", "chat_completion")
    requires_formatting = model_info.get("requires_formatting", False)

    model_max_tokens = model_info.get("max_tokens", 4096)
    max_tokens = min(max_tokens, model_max_tokens)

    logger.info("Selected model: %s (%s), task: %s, API method: %s, formatting: %s, max tokens: %d",
                model_key, model_id, task_type, api_method, requires_formatting, max_tokens)
    logger.info("Prompt length: %d chars", len(prompt))

    input_tokens = approx_token_count(prompt)
    logger.info("Estimated input tokens: %d", input_tokens)

    start_time = time.time()
    response_text = ""

    try:
        if api_method == "chat_completions_create":
            # Используем chat.completions.create как в примере
            logger.info(f"Using chat.completions.create for {model_id}")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            logger.info(f"Sending to {model_id} with messages: {messages}")

            completion = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stream=False
            )

            # Для SmolLM3 используем специальный парсер
            if model_key == "smollm3":
                response_text = extract_smollm3_response(completion)
            else:
                response_text = extract_response_from_completion(completion)

        elif api_method == "chat_completion":
            # Альтернативный метод для совместимости
            logger.info(f"Using chat_completion for {model_id}")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            out = client.chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stop=None
            )

            response_text = extract_response_from_completion(out)

        else:
            # Для text-generation моделей
            logger.info(f"Using text_generation for {model_id}")

            out = client.text_generation(
                model=model_id,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                stream=False
            )

            if hasattr(out, 'generated_text'):
                response_text = out.generated_text
            elif isinstance(out, str):
                response_text = out
            elif isinstance(out, dict) and "generated_text" in out:
                response_text = out["generated_text"]
            else:
                response_text = str(out)

        elapsed = time.time() - start_time
        logger.info("Model responded in %.3f sec", elapsed)

        logger.info(f"Raw response length: {len(response_text)}")
        logger.info(f"Raw response (first 200 chars): {response_text[:200]}")

        # Очищаем ответ (НО НЕ ОБРЕЗАЕМ ПО ПЕРЕНОСАМ СТРОК)
        response_text = cleanup_response(response_text, prompt, model_info)

        logger.info(f"After cleanup length: {len(response_text)}")
        logger.info(f"After cleanup (first 200 chars): {response_text[:200]}")

        # Логируем ответ
        if response_text:
            logger.info("Response received (%d chars)", len(response_text))
            if len(response_text) > 500:
                logger.info("Response (first 500 chars): %s...", response_text[:500])
        else:
            logger.warning("Empty response received")

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        logger.error("Model inference error after %.3f sec: %s", elapsed, error_msg)
        logger.error(traceback.format_exc())

        return jsonify({
            "error": f"Model {model_id} inference failed",
            "details": str(e),
            "model_key": model_key
        }), 500

    output_tokens = approx_token_count(response_text)
    total_tokens = input_tokens + output_tokens

    logger.info("Output tokens: %d, total: %d", output_tokens, total_tokens)

    cost = estimate_cost(input_tokens, output_tokens, model_info)

    result = {
        "model_key": model_key,
        "model_id": model_id,
        "model_provider": model_info["provider"],
        "task_type": task_type,
        "is_free": model_info.get("is_free", False),
        "prompt": prompt,
        "response_text": response_text,
        "time_s": round(elapsed, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 8) if cost is not None else None,
        "max_tokens_limit": model_info.get("max_tokens", 200),
        "pricing_info": get_model_display_info(model_key, model_info).get("pricing_info", "Unknown"),
        "description": model_info.get("description", ""),
        "model_size": model_info.get("model_size", "Unknown")
    }

    if "context_window" in model_info:
        result["context_window"] = model_info["context_window"]

    logger.info("Returning result with time: %.3fs, tokens: %d, cost: $%.8f",
                elapsed, total_tokens, cost or 0.0)
    return jsonify(result)


@app.route('/api/test_all_models', methods=['POST'])
def test_all_models():
    """Тестирование всех моделей одним запросом"""
    body = request.get_json() or {}
    prompt = body.get("prompt", "What is the capital of France?")
    max_tokens = body.get("max_tokens", 100)

    results = {}

    for model_key in MODEL_ORDER:
        if model_key not in MODELS:
            continue
        try:
            logger.info(f"Testing model: {model_key}")

            # Вызываем generate для каждой модели
            with app.test_request_context('/api/generate', method='POST',
                                          json={
                                              "model_key": model_key,
                                              "prompt": prompt,
                                              "max_tokens": max_tokens
                                          }):
                response = generate()

            if isinstance(response, tuple):
                response_data = response[0].get_json()
            else:
                response_data = response.get_json()

            if "error" not in response_data:
                results[model_key] = {
                    "success": True,
                    "response": response_data.get("response_text", ""),
                    "time_s": response_data.get("time_s", 0),
                    "tokens": response_data.get("total_tokens", 0),
                    "cost": response_data.get("estimated_cost_usd", 0),
                    "is_free": response_data.get("is_free", False),
                    "model_id": response_data.get("model_id", "")
                }
            else:
                results[model_key] = {
                    "success": False,
                    "error": response_data.get("error", "Unknown error"),
                    "details": response_data.get("details", "")
                }

        except Exception as e:
            results[model_key] = {
                "success": False,
                "error": str(e)[:100]
            }

    return jsonify({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "results": results,
        "total_models": len(MODELS),
        "successful_models": sum(1 for r in results.values() if r.get("success", False))
    })


@app.route('/api/test_smollm3_exact', methods=['GET'])
def test_smollm3_exact():
    """Точный тест SmolLM3 как в примере"""
    try:
        completion = client.chat.completions.create(
            model="HuggingFaceTB/SmolLM3-3B:hf-inference",
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France?"
                }
            ],
            max_tokens=100,
            temperature=0.7,
            stop=None
        )

        # Извлекаем ответ как в примере
        response = ""
        if hasattr(completion, 'choices') and completion.choices:
            message = completion.choices[0].message
            if hasattr(message, 'content') and message.content:
                response = message.content
            else:
                response = str(message)

        return jsonify({
            "success": True,
            "model": "HuggingFaceTB/SmolLM3-3B:hf-inference",
            "test_prompt": "What is the capital of France?",
            "response": response,
            "method": "completion.choices[0].message"
        })

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/model_stats', methods=['GET'])
def model_stats():
    """Статистика по всем моделям"""
    stats = {
        "total_models": len(MODELS),
        "models": {},
        "free_models": 0,
        "paid_models": 0
    }

    for key, info in MODELS.items():
        model_stat = {
            "id": info["id"],
            "provider": info["provider"],
            "is_free": info.get("is_free", False),
            "max_tokens": info.get("max_tokens", 200),
            "description": info.get("description", ""),
            "model_size": info.get("model_size", "Unknown")
        }

        if info.get("is_free", False):
            stats["free_models"] += 1
            model_stat["cost"] = "Free"
        else:
            stats["paid_models"] += 1
            if "input_price_per_mt" in info:
                model_stat["cost"] = f"${info['input_price_per_mt']}/1M input, ${info.get('output_price_per_mt', info['input_price_per_mt'])}/1M output"
            else:
                model_stat["cost"] = "Paid (unknown rate)"

        stats["models"][key] = model_stat

    return jsonify(stats)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logger.info("Starting server on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=True)