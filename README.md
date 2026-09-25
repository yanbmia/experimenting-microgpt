# experimenting-microgpt

Forked from [Karpathy's microGPT](https://github.com/karpathy/microGPT).


## 1. microgpt.py

```bash
python microgpt.py
```

Trains a tiny (~4k parameter) character-level GPT on a list of names (downloaded automatically on first run) and samples 20 new, made-up names. 

It's also importable as a class:

```python
from microgpt import MicroGPT

model = MicroGPT(n_layer=1, n_embd=16, block_size=16, n_head=4)
model.train(["emma", "olivia", "ava", ...], num_steps=1000)
print(model.generate(max_tokens=16, temperature=0.5))
```

Keep `block_size` and document lengths small. It learns texture/statistics of short strings. It cannot do reading comprehension/hold a conversation.

## 2. simple_chatbot.py — keyword Q&A + MicroGPT babble demo

```bash
python simple_chatbot.py sample_document.txt
```

Loads a document and lets you:
- Ask questions (answered via keyword/sentence matching against the document; no LLM involved)
- Type `summary` for an extractive summary
- Type `babble` to watch MicroGPT train for a few seconds on chunks of the loaded document, then hallucinate fake words/phrases from what it learned
- Type `quit` to exit

## 3. ollama_chatbot.py — LLM document Q&A

Backed by a real local language model through [Ollama](https://ollama.com), so it can answer open-ended questions accurately.

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

No third-party Python packages required. It talks to Ollama's local REST API (`localhost:11434`) directly. If Ollama isn't running or the model isn't pulled, it'll tell you exactly what to do instead of crashing.

Type `quit` to exit.
