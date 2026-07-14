#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import base64
import io
import json
import os
import sys
import time
import uuid
from typing import Optional

import numpy as np
import pymysql
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FASTER_QWEN_DIR = os.path.join(BASE_DIR, "faster-qwen3-tts")
if FASTER_QWEN_DIR not in sys.path:
    sys.path.insert(0, FASTER_QWEN_DIR)

from faster_qwen3_tts import FasterQwen3TTS


CONFIG_FILE = os.getenv("TTS_CONFIG_FILE", "tts_config.json")


def load_config():
    if not os.path.exists(CONFIG_FILE):
        return {}

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


APP_CONFIG = load_config()
MODEL_CONFIG = APP_CONFIG.get("model", {})
MYSQL_CONFIG = APP_CONFIG.get("mysql", {})

HOST = "0.0.0.0"
PORT = int(MODEL_CONFIG.get("faster_port", MODEL_CONFIG.get("port", 8093)))

MODEL_NAME = MODEL_CONFIG.get("name", "/home/wt/projects/QwenTTS/qwentts")
PROMPT_MODEL_ID = MODEL_CONFIG.get("prompt_model_id", MODEL_NAME)
DEVICE = MODEL_CONFIG.get("device", MODEL_CONFIG.get("device_map", "cuda:0"))
DTYPE_NAME = MODEL_CONFIG.get("dtype", "bfloat16")
ATTN_IMPLEMENTATION = MODEL_CONFIG.get("attention", "sdpa")
MAX_SEQ_LEN = int(MODEL_CONFIG.get("max_seq_len", 2048))

DEFAULT_LANGUAGE = MODEL_CONFIG.get("default_language", "Chinese")
SAMPLE_RATE = int(MODEL_CONFIG.get("sample_rate", 24000))
CHUNK_SIZE = int(MODEL_CONFIG.get("chunk_size", 4))
MAX_NEW_TOKENS = int(MODEL_CONFIG.get("max_new_tokens", 2048))
MIN_NEW_TOKENS = int(MODEL_CONFIG.get("min_new_tokens", 2))
TEMPERATURE = float(MODEL_CONFIG.get("temperature", 0.9))
TOP_K = int(MODEL_CONFIG.get("top_k", 50))
TOP_P = float(MODEL_CONFIG.get("top_p", 1.0))
DO_SAMPLE = bool(MODEL_CONFIG.get("do_sample", True))
REPETITION_PENALTY = float(MODEL_CONFIG.get("repetition_penalty", 1.05))
NON_STREAMING_MODE = MODEL_CONFIG.get("non_streaming_mode", None)
PARITY_MODE = bool(MODEL_CONFIG.get("parity_mode", False))
APPEND_SILENCE = bool(MODEL_CONFIG.get("append_silence", True))

CUDA_SYNC_TIMING = bool(MODEL_CONFIG.get("cuda_sync_timing", True))
ENABLE_GENERATION_LOCK = bool(MODEL_CONFIG.get("enable_generation_lock", True))
TORCH_NUM_THREADS = int(MODEL_CONFIG.get("torch_num_threads", 1))
TORCH_NUM_INTEROP_THREADS = int(MODEL_CONFIG.get("torch_num_interop_threads", 1))
ENABLE_TF32 = bool(MODEL_CONFIG.get("enable_tf32", True))
PERF_LOG_CHUNKS = bool(MODEL_CONFIG.get("perf_log_chunks", True))

MYSQL_HOST = MYSQL_CONFIG.get("host", "192.168.3.139")
MYSQL_PORT = int(MYSQL_CONFIG.get("port", 3307))
MYSQL_DATABASE = MYSQL_CONFIG.get("database", "xiaozhi_esp32_server")
MYSQL_USER = MYSQL_CONFIG.get("user", "root")
MYSQL_PASSWORD = MYSQL_CONFIG.get("password", "")
MYSQL_TABLE = MYSQL_CONFIG.get("table", "tts_voice_prompt")

app = FastAPI()

model: Optional[FasterQwen3TTS] = None
generation_lock = asyncio.Lock()
voice_prompt_cache = {}
voice_prompt_cache_lock = asyncio.Lock()


def now() -> float:
    return time.perf_counter()


def cuda_sync():
    if CUDA_SYNC_TIMING and torch.cuda.is_available():
        torch.cuda.synchronize()


def get_torch_dtype():
    if DTYPE_NAME == "float16":
        return torch.float16
    if DTYPE_NAME == "float32":
        return torch.float32
    return torch.bfloat16


def configure_torch_runtime():
    if TORCH_NUM_THREADS > 0:
        torch.set_num_threads(TORCH_NUM_THREADS)
    if TORCH_NUM_INTEROP_THREADS > 0:
        try:
            torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)
        except RuntimeError:
            pass
    if ENABLE_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def gpu_mem(prefix: str = "") -> str:
    if not torch.cuda.is_available():
        return f"{prefix}cuda=n/a"

    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    max_allocated = torch.cuda.max_memory_allocated() / 1024 / 1024
    return (
        f"{prefix}gpu_mem allocated={allocated:.1f}MB, "
        f"reserved={reserved:.1f}MB, max_allocated={max_allocated:.1f}MB"
    )


def float_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    audio = audio * 0.80
    audio = np.tanh(audio) * 0.90
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    return pcm.tobytes()


def get_mysql_connection():
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def ensure_voice_prompt_columns():
    sql = """
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
    """

    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (MYSQL_DATABASE, MYSQL_TABLE))
            columns = {row["COLUMN_NAME"] for row in cursor.fetchall()}
            if "model_name" not in columns:
                cursor.execute(
                    f"ALTER TABLE `{MYSQL_TABLE}` "
                    "ADD COLUMN `model_name` VARCHAR(1024) NOT NULL DEFAULT '' AFTER `name`"
                )
        conn.commit()


def load_voice_prompt_from_db(spk_id: str):
    sql = f"""
        SELECT prompt_blob
        FROM `{MYSQL_TABLE}`
        WHERE spk_id = %s AND model_name = %s AND enabled = 1
        LIMIT 1
    """

    with get_mysql_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(sql, (spk_id, PROMPT_MODEL_ID))
            row = cursor.fetchone()

    if not row:
        raise ValueError(
            f"voice prompt not found for spk_id={spk_id}, model_name={PROMPT_MODEL_ID}. "
            "Regenerate this voice prompt with the same model."
        )

    buffer = io.BytesIO(row["prompt_blob"])
    map_location = DEVICE if torch.cuda.is_available() else "cpu"
    try:
        return torch.load(buffer, map_location=map_location, weights_only=False)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location=map_location)


async def get_voice_prompt(spk_id: str):
    cached = voice_prompt_cache.get(spk_id)
    if cached is not None:
        return cached

    async with voice_prompt_cache_lock:
        cached = voice_prompt_cache.get(spk_id)
        if cached is not None:
            return cached

        prompt = await asyncio.to_thread(load_voice_prompt_from_db, spk_id)
        voice_prompt_cache[spk_id] = prompt
        print(f"[voice] loaded voice_clone_prompt from mysql: spk_id={spk_id}")
        return prompt


def next_generator_item(generator, session_id: str):
    try:
        return False, next(generator)
    except StopIteration:
        return True, None
    except Exception as exc:
        print(f"[ws] generator error session={session_id}: {repr(exc)}")
        raise


def build_stream_kwargs(text, language, voice_clone_prompt, chunk_size):
    return {
        "text": text,
        "language": language,
        "voice_clone_prompt": voice_clone_prompt,
        "ref_text": "",
        "ref_audio": None,
        "chunk_size": chunk_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_new_tokens": MIN_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "top_p": TOP_P,
        "do_sample": DO_SAMPLE,
        "repetition_penalty": REPETITION_PENALTY,
        "non_streaming_mode": NON_STREAMING_MODE,
        "parity_mode": PARITY_MODE,
        "append_silence": APPEND_SILENCE,
    }


@app.on_event("startup")
def startup_load_model():
    global model, SAMPLE_RATE

    configure_torch_runtime()

    print("[startup] faster-qwen3-tts model:", MODEL_NAME)
    print("[startup] prompt_model_id:", PROMPT_MODEL_ID)
    print("[startup] device:", DEVICE)
    print("[startup] dtype:", DTYPE_NAME)
    print("[startup] attention:", ATTN_IMPLEMENTATION)
    print("[startup] max_seq_len:", MAX_SEQ_LEN)
    print("[startup] chunk_size:", CHUNK_SIZE)
    print("[startup] mysql:", f"{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}.{MYSQL_TABLE}")
    print("[startup] cuda_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[startup] gpu:", torch.cuda.get_device_name(0))
        print("[startup] torch cuda:", torch.version.cuda)

    ensure_voice_prompt_columns()

    t0 = now()
    model = FasterQwen3TTS.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        dtype=get_torch_dtype(),
        attn_implementation=ATTN_IMPLEMENTATION,
        max_seq_len=MAX_SEQ_LEN,
    )
    SAMPLE_RATE = int(getattr(model, "sample_rate", SAMPLE_RATE))
    cuda_sync()
    print(f"[startup] model loaded: {now() - t0:.3f}s")
    print("[startup]", gpu_mem())


@app.get("/")
def index():
    return {
        "service": "faster-qwen3-tts paddlespeech compatible tts",
        "model": MODEL_NAME,
        "prompt_model_id": PROMPT_MODEL_ID,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "attention": ATTN_IMPLEMENTATION,
        "voice_source": "mysql",
        "voice_table": MYSQL_TABLE,
        "samplerate": "/paddlespeech/tts/streaming/samplerate",
        "websocket": "/paddlespeech/tts/streaming",
        "sample_rate": SAMPLE_RATE,
        "chunk_size": CHUNK_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "parity_mode": PARITY_MODE,
    }


@app.get("/paddlespeech/tts/streaming/samplerate")
def get_samplerate():
    return {"sample_rate": SAMPLE_RATE}


@app.websocket("/paddlespeech/tts/streaming")
@app.websocket("/paddlespeech/tts/streaming/")
async def paddlespeech_compatible_ws(websocket: WebSocket):
    await websocket.accept()

    session_id = str(uuid.uuid4())
    started = False
    conn_t0 = now()
    current_generation_task = None
    current_cancel_event = None
    send_lock = asyncio.Lock()

    print(f"[ws] connected: {session_id}")

    async def send_json(payload):
        async with send_lock:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))

    def cleanup_done_generation():
        nonlocal current_generation_task, current_cancel_event
        if current_generation_task is None or not current_generation_task.done():
            return
        try:
            current_generation_task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[ws] generation task error: {repr(e)}")
        current_generation_task = None
        current_cancel_event = None

    async def cancel_current_generation(reason: str):
        nonlocal current_generation_task, current_cancel_event
        if current_cancel_event is not None:
            current_cancel_event.set()

        if current_generation_task is not None and not current_generation_task.done():
            print(f"[ws] cancel generation session={session_id}, reason={reason}")
            try:
                await current_generation_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[ws] cancel generation error: {repr(e)}")

        current_generation_task = None
        current_cancel_event = None
        await send_json({
            "status": 0,
            "signal": "cancel",
            "session": session_id,
            "reason": reason,
        })

    async def run_generation(text, spk_id, language, chunk_size, cancel_event):
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            total_t0 = now()
            selected_voice_prompt = await get_voice_prompt(str(spk_id))

            if cancel_event.is_set():
                raise asyncio.CancelledError()

            generator = model.generate_voice_clone_streaming(
                **build_stream_kwargs(
                    text=text,
                    language=language,
                    voice_clone_prompt=selected_voice_prompt,
                    chunk_size=chunk_size,
                )
            )

            chunk_count = 0
            total_samples = 0
            total_pcm_bytes = 0
            first_chunk_time = None
            prev_chunk_t = total_t0
            total_pcm_time = 0.0
            total_b64_time = 0.0
            total_ws_send_time = 0.0

            while True:
                if cancel_event.is_set():
                    raise asyncio.CancelledError()

                next_t0 = now()
                is_done, item = await asyncio.to_thread(next_generator_item, generator, session_id)
                next_t1 = now()
                if is_done:
                    break

                chunk, sr, timing = item

                if first_chunk_time is None:
                    first_chunk_time = next_t1 - total_t0
                    print(
                        f"[perf] session={session_id} first_chunk={first_chunk_time:.3f}s, "
                        f"sr={sr}, timing={timing}"
                    )

                pcm_t0 = now()
                pcm_bytes = float_to_pcm16_bytes(chunk)
                pcm_t1 = now()
                total_pcm_time += pcm_t1 - pcm_t0

                b64_t0 = now()
                audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                b64_t1 = now()
                total_b64_time += b64_t1 - b64_t0

                if cancel_event.is_set():
                    raise asyncio.CancelledError()

                send_t0 = now()
                await send_json({
                    "status": 1,
                    "audio": audio_b64,
                    "session": session_id,
                })
                send_t1 = now()
                total_ws_send_time += send_t1 - send_t0

                chunk_count += 1
                samples = int(len(chunk))
                total_samples += samples
                total_pcm_bytes += len(pcm_bytes)

                if PERF_LOG_CHUNKS:
                    audio_sec = samples / sr if sr else 0
                    print(
                        f"[chunk] session={session_id} idx={chunk_count} "
                        f"samples={samples} audio={audio_sec:.3f}s "
                        f"interval={next_t1 - prev_chunk_t:.3f}s "
                        f"next={next_t1 - next_t0:.3f}s "
                        f"pcm={pcm_t1 - pcm_t0:.4f}s "
                        f"b64={b64_t1 - b64_t0:.4f}s "
                        f"ws_send={send_t1 - send_t0:.4f}s "
                        f"timing={timing}"
                    )

                prev_chunk_t = send_t1

            await send_json({"status": 2, "session": session_id})
            total_t1 = now()
            audio_seconds = total_samples / SAMPLE_RATE if SAMPLE_RATE else 0
            total_time = total_t1 - total_t0
            rtf = total_time / audio_seconds if audio_seconds > 0 else 0

            print(
                f"[summary] session={session_id} chars={len(text)} chunks={chunk_count} "
                f"audio={audio_seconds:.3f}s total={total_time:.3f}s rtf={rtf:.3f} "
                f"pcm_total={total_pcm_time:.4f}s b64_total={total_b64_time:.4f}s "
                f"ws_send_total={total_ws_send_time:.4f}s pcm_bytes={total_pcm_bytes}"
            )
            print("[summary]", gpu_mem(prefix=f"session={session_id} "))

        except asyncio.CancelledError:
            print(f"[ws] generation cancelled: session={session_id}")
            try:
                await send_json({
                    "status": 3,
                    "signal": "cancelled",
                    "session": session_id,
                })
            except Exception:
                pass
            raise
        except Exception as e:
            print("[ws] error:", repr(e))
            await send_json({
                "status": -1,
                "message": repr(e),
                "session": session_id,
            })

    try:
        while True:
            cleanup_done_generation()

            recv_wait_t0 = now()
            msg = await websocket.receive_text()
            recv_wait_t1 = now()

            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                await send_json({
                    "status": -1,
                    "message": "invalid json",
                    "session": session_id,
                })
                continue

            if data.get("task") == "tts" and data.get("signal") == "start":
                started = True
                send_t0 = now()
                await send_json({"status": 0, "session": session_id})
                send_t1 = now()
                print(
                    f"[ws] start session={session_id}, "
                    f"recv_wait={recv_wait_t1 - recv_wait_t0:.4f}s, "
                    f"send_ack={send_t1 - send_t0:.4f}s"
                )
                continue

            if data.get("task") == "tts" and data.get("signal") == "cancel":
                await cancel_current_generation("client_cancel")
                continue

            if data.get("task") == "tts" and data.get("signal") == "end":
                recv_session = data.get("session") or session_id
                if current_generation_task is not None and not current_generation_task.done():
                    await cancel_current_generation("client_end")

                await send_json({
                    "status": 0,
                    "signal": "end",
                    "session": recv_session,
                })
                print(f"[ws] end session={recv_session}, conn_total={now() - conn_t0:.3f}s")
                break

            request_parse_t0 = now()
            text = data.get("text", "").strip()

            if not text:
                await send_json({
                    "status": -1,
                    "message": "text is required",
                    "session": session_id,
                })
                continue

            if not started:
                await send_json({
                    "status": -1,
                    "message": "please send start signal first",
                    "session": session_id,
                })
                continue

            if current_generation_task is not None and not current_generation_task.done():
                await send_json({
                    "status": -1,
                    "message": "tts generation is busy",
                    "session": session_id,
                })
                continue

            spk_id = data.get("spk_id", 0)
            language = data.get("language", DEFAULT_LANGUAGE)
            chunk_size = int(data.get("chunk_size", CHUNK_SIZE))
            request_parse_t1 = now()

            print(
                f"[ws] tts request session={session_id}, spk_id={spk_id}, "
                f"language={language}, chars={len(text)}, chunk_size={chunk_size}, text={text}"
            )
            print(
                f"[perf] session={session_id} request_parse={request_parse_t1 - request_parse_t0:.4f}s, "
                f"recv_wait={recv_wait_t1 - recv_wait_t0:.4f}s"
            )

            queue_t0 = now()
            current_cancel_event = asyncio.Event()

            async def generation_runner():
                if ENABLE_GENERATION_LOCK:
                    async with generation_lock:
                        print(f"[perf] session={session_id} queue_wait={now() - queue_t0:.4f}s")
                        await run_generation(text, spk_id, language, chunk_size, current_cancel_event)
                else:
                    print(f"[perf] session={session_id} queue_wait=0.0000s lock_disabled")
                    await run_generation(text, spk_id, language, chunk_size, current_cancel_event)

            current_generation_task = asyncio.create_task(generation_runner())

    except WebSocketDisconnect:
        print(f"[ws] disconnected: {session_id}")
    except Exception as e:
        print(f"[ws] fatal error: {repr(e)}")
        try:
            await send_json({
                "status": -1,
                "message": repr(e),
                "session": session_id,
            })
        except Exception:
            pass
    finally:
        if current_generation_task is not None and not current_generation_task.done():
            if current_cancel_event is not None:
                current_cancel_event.set()
        print(f"[ws] finished: {session_id}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
