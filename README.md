# experimenting-microgpt

Forked from [Karpathy's microGPT](https://github.com/karpathy/microGPT) — the most atomic, dependency-free implementation of a GPT in pure Python. This repo extends it with two small chatbot demos.

There are three tools here:

1. **`microgpt.py`** — the from-scratch, pure-Python GPT (autograd engine + transformer, no numpy/torch). Good for understanding how a GPT actually works under the hood. Too small to be "smart."
2. **`simple_chatbot.py`** — loads a text file and answers questions about it using keyword matching, plus a `babble` command that trains MicroGPT live on the document and shows what a tiny char-level model actually produces.
3. **`ollama_chatbot.py`** — loads a text file and answers questions about it using a real local LLM via [Ollama](https://ollama.com). This is the "actually useful" version.

## 1. microgpt.py — the toy GPT itself

```bash
python microgpt.py
```

Trains a tiny (~4k parameter) character-level GPT on a list of names (downloaded automatically on first run) and samples 20 new, made-up names. Everything — autograd, attention, Adam optimizer — is implemented from scratch in plain Python, no dependencies.

It's also importable as a class:

```python
from microgpt import MicroGPT

model = MicroGPT(n_layer=1, n_embd=16, block_size=16, n_head=4)
model.train(["emma", "olivia", "ava", ...], num_steps=1000)
print(model.generate(max_tokens=16, temperature=0.5))
```

Keep `block_size` and document lengths small — this is a teaching tool, not a production model. It learns texture/statistics of short strings; it cannot do reading comprehension or hold a conversation.

## 2. simple_chatbot.py — keyword Q&A + MicroGPT babble demo

```bash
python simple_chatbot.py sample_document.txt
```

Loads a document and lets you:
- Ask questions (answered via keyword/sentence matching against the document — no LLM involved)
- Type `summary` for an extractive summary
- Type `babble` to watch MicroGPT train for a few seconds on chunks of the loaded document, then hallucinate fake words/phrases from what it learned — a fun way to see what a tiny char-level GPT actually does
- Type `quit` to exit

## 3. ollama_chatbot.py — real LLM document Q&A

This is the practical version: same "load a doc, ask questions" interface, but backed by a real local language model through [Ollama](https://ollama.com), so it can actually answer open-ended questions accurately.

### Setup
1. Install Ollama: https://ollama.com/download
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Make sure Ollama is running (the installer usually starts it automatically as a background app; if not, run `ollama serve`).

### Run
```bash
python ollama_chatbot.py sample_document.txt
python ollama_chatbot.py sample_document.txt --model llama3.2
```

No third-party Python packages required — it talks to Ollama's local REST API (`localhost:11434`) directly. If Ollama isn't running or the model isn't pulled, it'll tell you exactly what to do instead of crashing.

Type `quit` to exit.
