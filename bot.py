import os
import re
import json
import time
import io
import threading
import contextlib
import urllib.request
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from openai import OpenAI

# ---------------------------------------------------------
# Configuration & Environment Variables
# ---------------------------------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN") or os.environ.get("TELEGRAM_BOT_TOKEN")
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

API_KEY = OPENROUTER_API_KEY or AIPIPE_TOKEN
API_BASE_URL = os.environ.get("MODEL_BASE_URL") or ("https://openrouter.ai/api/v1" if OPENROUTER_API_KEY else "https://aipipe.org/openai/v1")
MODEL_NAME = os.environ.get("MODEL") or os.environ.get("MODEL_NAME") or "gpt-5-mini"
# Clean BASE_URL fallback to avoid stray characters or trailing slashes
BASE_URL = (
    os.environ.get("BASE_URL") 
    or os.environ.get("RENDER_EXTERNAL_URL") 
    or "https://tds-databot-n2ib.onrender.com"
).strip().rstrip("/")

LOG_URL = f"{BASE_URL}/run.jsonl"
LOG_FILE_PATH = "run.jsonl"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

app = FastAPI()

# Store chat history in memory per chat_id
chat_histories = {}

def write_log(entry: dict):
    """Appends a log entry to run.jsonl."""
    try:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Logging error: {e}")

# ---------------------------------------------------------
# FastAPI Endpoints
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "time": time.time()}

@app.get("/run.jsonl")
def get_log():
    if os.path.exists(LOG_FILE_PATH):
        return FileResponse(LOG_FILE_PATH, media_type="application/x-ndjson")
    return JSONResponse(content={"error": "Log file not found"}, status_code=404)

# ---------------------------------------------------------
# Python Execution Tool for Agent
# ---------------------------------------------------------
def run_python(code: str) -> str:
    """Executes Python code safely, capturing stdout."""
    output_capture = io.StringIO()
    global_scope = {
        "pd": pd,
        "np": np,
        "requests": requests,
        "bs4": BeautifulSoup,
        "json": json,
        "re": re,
    }
    try:
        with contextlib.redirect_stdout(output_capture):
            exec(code, global_scope)
        res = output_capture.getvalue()
        return res if res.strip() else "(Execution completed with no stdout)"
    except Exception as e:
        return f"Python Execution Error: {str(e)}"

tools = [{
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Execute Python code to fetch, clean, parse, or compute data (pandas, numpy, requests, bs4 available). Print final results to stdout.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python executable code."}
            },
            "required": ["code"]
        }
    }
}]

# ---------------------------------------------------------
# Agent Intelligence & Multi-Turn Processing
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are an expert Data Analyst Telegram Bot.
Your goal is to answer data questions accurately.

CRITICAL OUTPUT RULES:
1. You MUST respond with ONLY a single valid JSON object.
2. NO markdown code fences (do NOT use ```json ... ```), NO introductory prose, NO trailing explanation.
3. The JSON MUST follow this exact top-level schema:
   {"answer": <answer_content>, "log_url": "<PLACEHOLDER>"}
4. The <answer_content> must strictly match the exact shape/structure requested by the question.
   - If the user sends setup info ("I will send data next"), reply with an acknowledgement JSON:
     {"answer": "Awaiting data.", "log_url": "<PLACEHOLDER>"}
5. Use the `run_python` tool to fetch public datasets (MOSPI, SRS, etc.) or compute statistics when needed. Never guess a calculation.
"""

def extract_valid_json(text: str) -> dict:
    """Cleans prose/fences and extracts the target JSON object."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data["log_url"] = LOG_URL
            return data
    except Exception:
        pass
    
    return {"answer": text.strip(), "log_url": LOG_URL}

def process_message(chat_id: int, user_text: str) -> dict:
    start_time = time.time()
    
    if chat_id not in chat_histories:
        chat_histories[chat_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    history = chat_histories[chat_id]
    history.append({"role": "user", "content": user_text})
    
    # Cap conversation history length
    if len(history) > 20:
        history = [history[0]] + history[-19:]
    
    iterations = 0
    max_iterations = 10
    final_json = None
    
    while iterations < max_iterations:
        iterations += 1
        # Enforce budget cutoff at 200 seconds
        if time.time() - start_time > 200:
            break
            
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=history,
                tools=tools,
                temperature=0.0
            )
            msg = response.choices[0].message
            history.append(msg.model_dump(exclude_unset=True))
            
            if not msg.tool_calls:
                final_json = extract_valid_json(msg.content or "")
                break
                
            for tool_call in msg.tool_calls:
                if tool_call.function.name == "run_python":
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                    output = run_python(code)
                    
                    write_log({
                        "timestamp": time.time(),
                        "chat_id": chat_id,
                        "executed_code": code,
                        "output": output[:2000]
                    })
                    
                    history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": "run_python",
                        "content": output[:8000]
                    })
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            clean_input = user_text.lower().strip()
            if clean_input in ["/start", "hi", "hii", "hello"]:
                final_json = {"answer": "Ready for data questions.", "log_url": LOG_URL}
            else:
                final_json = {"answer": f"Error: {str(e)}", "log_url": LOG_URL}
            break
            
    if not final_json:
        final_json = {"answer": "Unable to compute answer within timeout.", "log_url": LOG_URL}
        
    write_log({
        "timestamp": time.time(),
        "chat_id": chat_id,
        "question": user_text,
        "final_reply": final_json
    })
    
    return final_json

# ---------------------------------------------------------
# Telegram Long-Polling Loop
# ---------------------------------------------------------
def telegram_polling():
    if not BOT_TOKEN:
        print("BOT_TOKEN missing! Telegram polling disabled.")
        return

    offset = 0
    tg_api = f"https://api.telegram.org/bot{BOT_TOKEN}"
    print("Started Telegram long-polling loop...")

    while True:
        try:
            url = f"{tg_api}/getUpdates?offset={offset}&timeout=30"
            resp = requests.get(url, timeout=35).json()
            
            if resp.get("ok"):
                for result in resp.get("result", []):
                    offset = result["update_id"] + 1
                    msg = result.get("message")
                    if not msg or "text" not in msg:
                        continue
                        
                    chat_id = msg["chat"]["id"]
                    text = msg["text"]
                    
                    print(f"Received msg from {chat_id}: {text}")
                    
                    # Compute answer
                    reply_dict = process_message(chat_id, text)
                    reply_text = json.dumps(reply_dict)
                    
                    # Send response back to Telegram
                    requests.post(f"{tg_api}/sendMessage", json={
                        "chat_id": chat_id,
                        "text": reply_text
                    })
        except Exception as e:
            print(f"Telegram polling error: {e}")
            time.sleep(3)

# ---------------------------------------------------------
# Self-Ping Keep-Alive Thread
# ---------------------------------------------------------
def keep_alive():
    while True:
        time.sleep(600)  # Ping every 10 mins
        if BASE_URL and "localhost" not in BASE_URL:
            try:
                urllib.request.urlopen(f"{BASE_URL}/health", timeout=10)
                print("Self-ping keep-alive successful.")
            except Exception as e:
                print(f"Self-ping failed: {e}")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=telegram_polling, daemon=True).start()
    threading.Thread(target=keep_alive, daemon=True).start()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("bot:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
