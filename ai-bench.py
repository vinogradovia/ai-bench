import os
import sys
import time
from openai import OpenAI, APIConnectionError, APITimeoutError, AuthenticationError
import tiktoken

BASE_URL = os.getenv("AI_BENCH_BASE_URL", "https://localhost:8080/v1/")
API_KEY = os.getenv("AI_BENCH_API_KEY", "sk-***")
MODEL_NAME = os.getenv("AI_BENCH_MODEL_NAME", "local-default")
PROMPT = os.getenv("AI_BENCH_PROMPT", "Напиши подробное эссе на тему развития искусственного интеллекта до 2030 года.")


def validate_config():
    errors = []

    if not BASE_URL or BASE_URL.strip() == "":
        errors.append("AI_BENCH_BASE_URL не задан или пустой")

    if not API_KEY or API_KEY == "sk-***":
        errors.append("AI_BENCH_API_KEY не задан (используется дефолтное значение)")

    if not MODEL_NAME or MODEL_NAME.strip() == "":
        errors.append("AI_BENCH_MODEL_NAME не задан или пустой")

    if errors:
        print("=== ОШИБКИ КОНФИГУРАЦИИ ===")
        for e in errors:
            print(f"  - {e}")
        print()
        print("Необходимо задать переменные окружения:")
        print("  export AI_BENCH_BASE_URL=<url>")
        print("  export AI_BENCH_API_KEY=<key>")
        print("  export AI_BENCH_MODEL_NAME=<model>")
        print()
        sys.exit(1)


def test_non_streaming(client):
    print("=== ТЕСТ 1: нестриминг (stream=False) ===")
    try:
        start_time = time.time()

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.7,
            stream=False,
        )

        end_time = time.time()
        duration = end_time - start_time
        generated_text = response.choices[0].message.content

        enc = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(enc.encode(generated_text))
        tokens_per_second = num_tokens / duration

        print(f"Время выполнения: {duration:.2f} сек")
        print(f"Сгенерировано токенов: {num_tokens}")
        print(f"Чистая скорость API: {tokens_per_second:.2f} токенов/сек")
    except AuthenticationError:
        print("ОШИБКА: недействительный API-ключ")
        sys.exit(1)
    except APIConnectionError as e:
        print(f"ОШИБКА: не удалось подключиться к {BASE_URL}")
        print(f"  {e}")
        sys.exit(1)
    except APITimeoutError:
        print("ОШИБКА: превышено время ожидания ответа")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: {type(e).__name__}: {e}")
        sys.exit(1)
    print()


def test_streaming(client):
    print("=== ТЕСТ 2: стриминг (stream=True) ===")
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.7,
            stream=True,
        )

        first_token_time = None
        token_count = 0

        print("Генерация пошла...", end="", flush=True)

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1

        end_time = time.time()
        generation_time = end_time - first_token_time

        if generation_time == 0:
            print("\nОшибка: время генерации равно 0")
            sys.exit(1)

        print(f"\n\nЧистое время генерации текста: {generation_time:.2f} сек")
        print(f"Приблизительно токенов: {token_count}")
        print(f"Реальная скорость генерации API: {token_count / generation_time:.2f} токенов/сек")
    except AuthenticationError:
        print("ОШИБКА: недействительный API-ключ")
        sys.exit(1)
    except APIConnectionError as e:
        print(f"ОШИБКА: не удалось подключиться к {BASE_URL}")
        print(f"  {e}")
        sys.exit(1)
    except APITimeoutError:
        print("ОШИБКА: превышено время ожидания ответа")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА: {type(e).__name__}: {e}")
        sys.exit(1)


def main():
    validate_config()

    print(f"Отправка запроса к модели {MODEL_NAME}...")
    print(f"Base URL: {BASE_URL}\n")

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    test_non_streaming(client)
    test_streaming(client)

    print("\n=== БЕНЧМАРК ЗАВЕРШЕН ===")


if __name__ == "__main__":
    main()
