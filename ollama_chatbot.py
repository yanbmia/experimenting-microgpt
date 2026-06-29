"""
Document Q&A chatbot backed by a real local LLM via Ollama.

This is the "actually useful" counterpart to simple_chatbot.py. Where
simple_chatbot.py uses MicroGPT (a tiny, dependency-free, character-level toy
GPT that cannot do reading comprehension) plus keyword matching, this script
sends the document and your question to a real local language model running
in Ollama and gets a genuine answer back.

Setup:
    1. Install Ollama: https://ollama.com/download
    2. Pull a model, e.g.:   ollama pull llama3.2
    3. Make sure Ollama is running (it starts a local server on :11434;
       `ollama serve` if it's not already running in the background)

Usage:
    python ollama_chatbot.py sample_document.txt
    python ollama_chatbot.py sample_document.txt --model llama3.2
    python ollama_chatbot.py        # will prompt for a file path

No third-party Python packages required -- this talks to Ollama's local
REST API directly over plain urllib.
"""

import sys
import os
import json
import argparse
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2"


def load_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def check_ollama_running():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def list_models():
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            data = json.loads(resp.read())
            return [m['name'] for m in data.get('models', [])]
    except Exception:
        return []


def ollama_chat(model, messages, stream_callback=None):
    """
    Call Ollama's /api/chat endpoint. Streams tokens if stream_callback is
    given (called with each text chunk as it arrives); otherwise returns the
    full response string at once.
    """
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": stream_callback is not None,
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if stream_callback is None:
                body = json.loads(resp.read())
                return body.get("message", {}).get("content", "")

            full = []
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    stream_callback(piece)
                    full.append(piece)
            return "".join(full)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} ({e}). "
            "Is Ollama installed and running? Try `ollama serve`."
        )


def build_system_prompt(document_text):
    return (
        "You are a helpful assistant answering questions about a specific document. "
        "Only use information from the document below to answer. If the answer isn't "
        "in the document, say so clearly instead of making something up.\n\n"
        "--- DOCUMENT START ---\n"
        f"{document_text}\n"
        "--- DOCUMENT END ---"
    )


def main():
    parser = argparse.ArgumentParser(description="Chat with a document using a local Ollama model.")
    parser.add_argument("filepath", nargs="?", help="Path to the text file to load")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-stream", action="store_true", help="Disable token streaming")
    args = parser.parse_args()

    print("=" * 70)
    print("  Document Q&A Chatbot (powered by Ollama)")
    print("=" * 70)

    if not check_ollama_running():
        print("\n[!] Can't reach Ollama at http://localhost:11434.")
        print("    Install it from https://ollama.com/download and run `ollama serve`,")
        print("    or just open the Ollama app, then try again.")
        sys.exit(1)

    available = list_models()
    if available and args.model not in available and not any(args.model in m for m in available):
        print(f"\n[!] Model '{args.model}' not found locally. Available models: {', '.join(available)}")
        print(f"    Pull it with:  ollama pull {args.model}")
        sys.exit(1)

    filepath = args.filepath or input("\n[?] Enter path to text file: ").strip()
    content = load_file(filepath)
    if content is None:
        return

    print(f"\n[+] Loaded {len(content)} characters from '{filepath}'")
    print(f"[+] Using model: {args.model}")

    messages = [{"role": "system", "content": build_system_prompt(content)}]

    print("\n" + "=" * 70)
    print("  Ask anything about the document. Type 'quit' to exit.")
    print("=" * 70)

    while True:
        user_input = input("\n[You]: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("[*] Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        print("\n[Bot]: ", end="", flush=True)

        try:
            if args.no_stream:
                reply = ollama_chat(args.model, messages)
                print(reply)
            else:
                reply = ollama_chat(args.model, messages, stream_callback=lambda chunk: print(chunk, end="", flush=True))
                print()
        except RuntimeError as e:
            print(f"\n[Error] {e}")
            messages.pop()  # drop the failed turn so context stays clean
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
