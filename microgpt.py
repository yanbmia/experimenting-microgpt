"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

Forked from @karpathy's microGPT. The core autograd engine and transformer forward
pass are unchanged; this fork wraps the original script in a `MicroGPT` class so it
can be trained on arbitrary text (not just newline-separated names) and imported by
other scripts (see simple_chatbot.py).
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle


# ---------------------------------------------------------------------------
# Autograd: recursively apply the chain rule through a computation graph
# ---------------------------------------------------------------------------
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')  # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# ---------------------------------------------------------------------------
# Model building blocks (GPT-2-ish: layernorm -> rmsnorm, no biases, GeLU -> ReLU)
# ---------------------------------------------------------------------------
def matrix(nout, nin, std=0.08):
    return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


class MicroGPT:
    """
    A tiny, dependency-free, character-level GPT.

    This is a teaching toy, not a useful text generator: with a few thousand
    parameters and a tiny context window it can learn the *texture* of a small
    corpus (e.g. babble plausible-looking names, or echo short, very-repeated
    phrases from a document) but it cannot hold a real conversation or answer
    open-ended questions. For that, see ollama_chatbot.py in this repo, which
    uses a real local LLM via Ollama.
    """

    def __init__(self, n_layer=1, n_embd=16, block_size=16, n_head=4, seed=42):
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.seed = seed
        self.uchars = None
        self.vocab_size = None
        self.BOS = None
        self.state_dict = None
        self.params = None

    # -- tokenizer -----------------------------------------------------
    def _build_tokenizer(self, docs):
        uchars = sorted(set(''.join(docs)))  # unique characters in the dataset become token ids 0..n-1
        self.BOS = len(uchars)                # token id for a special Beginning of Sequence (BOS) token
        self.uchars = uchars
        self.vocab_size = len(uchars) + 1     # total number of unique tokens, +1 is for BOS

    def _encode(self, doc):
        return [self.BOS] + [self.uchars.index(ch) for ch in doc] + [self.BOS]

    # -- model init ------------------------------------------------------
    def _init_params(self):
        random.seed(self.seed)
        n_embd, vocab_size, block_size = self.n_embd, self.vocab_size, self.block_size
        state_dict = {
            'wte': matrix(vocab_size, n_embd),
            'wpe': matrix(block_size, n_embd),
            'lm_head': matrix(vocab_size, n_embd),
        }
        for i in range(self.n_layer):
            state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
            state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
        self.state_dict = state_dict
        self.params = [p for mat in state_dict.values() for row in mat for p in row]

    # -- forward pass ------------------------------------------------------
    def _forward(self, token_id, pos_id, keys, values):
        sd = self.state_dict
        tok_emb = sd['wte'][token_id]
        pos_emb = sd['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(self.n_layer):
            # 1) Multi-head Attention block
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, sd[f'layer{li}.attn_wq'])
            k = linear(x, sd[f'layer{li}.attn_wk'])
            v = linear(x, sd[f'layer{li}.attn_wv'])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs:hs + self.head_dim]
                k_h = [ki[hs:hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + self.head_dim] for vi in values[li]]
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / self.head_dim**0.5 for t in range(len(k_h))]
                attn_weights = softmax(attn_logits)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(self.head_dim)]
                x_attn.extend(head_out)
            x = linear(x_attn, sd[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]
            # 2) MLP block
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, sd[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, sd[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, sd['lm_head'])
        return logits

    # -- public API ------------------------------------------------------
    def train(self, docs, num_steps=1000, learning_rate=0.01, beta1=0.85, beta2=0.99,
              eps_adam=1e-8, verbose=True):
        """Train on a list of short documents/strings (each must fit in block_size)."""
        docs = [d for d in docs if d.strip()]
        if not docs:
            raise ValueError("docs must contain at least one non-empty string")
        random.seed(self.seed)
        random.shuffle(docs)
        self._build_tokenizer(docs)
        self._init_params()

        if verbose:
            print(f"num docs: {len(docs)}")
            print(f"vocab size: {self.vocab_size}")
            print(f"num params: {len(self.params)}")

        m = [0.0] * len(self.params)
        v = [0.0] * len(self.params)

        for step in range(num_steps):
            doc = docs[step % len(docs)]
            tokens = self._encode(doc)
            n = min(self.block_size, len(tokens) - 1)

            keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
            losses = []
            for pos_id in range(n):
                token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
                logits = self._forward(token_id, pos_id, keys, values)
                probs = softmax(logits)
                loss_t = -probs[target_id].log()
                losses.append(loss_t)
            loss = (1 / n) * sum(losses)

            loss.backward()

            lr_t = learning_rate * (1 - step / num_steps)
            for i, p in enumerate(self.params):
                m[i] = beta1 * m[i] + (1 - beta1) * p.grad
                v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
                m_hat = m[i] / (1 - beta1 ** (step + 1))
                v_hat = v[i] / (1 - beta2 ** (step + 1))
                p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
                p.grad = 0

            if verbose:
                print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')
        if verbose:
            print()
        return self

    def generate(self, max_tokens=16, temperature=0.5, prompt=None):
        """
        Sample a single string of up to `max_tokens` characters from the model.

        `prompt` (optional) seeds generation: the model is fed the prompt's
        characters first (any not in the training vocabulary are skipped),
        then samples continuation characters autoregressively. This is the
        method simple_chatbot.py calls; given this model's tiny size and
        char-level training, treat its output as illustrative, not factual.
        """
        if self.state_dict is None:
            raise RuntimeError("Model has not been trained yet. Call .train(docs) first.")

        keys, values = [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]
        token_id = self.BOS
        sample = []
        pos_id = 0

        if prompt:
            for ch in prompt:
                if pos_id >= self.block_size - 1:
                    break
                if ch not in self.uchars:
                    continue
                self._forward(token_id, pos_id, keys, values)
                token_id = self.uchars.index(ch)
                pos_id += 1

        while pos_id < self.block_size and len(sample) < max_tokens:
            logits = self._forward(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])
            token_id = random.choices(range(self.vocab_size), weights=[p.data for p in probs])[0]
            pos_id += 1
            if token_id == self.BOS:
                break
            sample.append(self.uchars[token_id])
        return ''.join(sample)


def _chunk_text(text, chunk_size):
    """Split text into block_size-friendly chunks (by whitespace, falling back to hard cuts)."""
    chunks = []
    words = text.split()
    current = ""
    for w in words:
        candidate = (current + " " + w).strip()
        if len(candidate) > chunk_size:
            if current:
                chunks.append(current)
            current = w[:chunk_size]
        else:
            current = candidate
    if current:
        chunks.append(current)
    return [c for c in chunks if c]


if __name__ == "__main__":
    # Reproduce the original Karpathy demo: train on names, babble new ones.
    random.seed(42)
    if not os.path.exists('input.txt'):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(names_url, 'input.txt')
    docs = [line.strip() for line in open('input.txt') if line.strip()]

    model = MicroGPT(n_layer=1, n_embd=16, block_size=16, n_head=4, seed=42)
    model.train(docs, num_steps=1000, learning_rate=0.01)

    print("--- inference (new, hallucinated names) ---")
    for sample_idx in range(20):
        sample = model.generate(max_tokens=model.block_size, temperature=0.5)
        print(f"sample {sample_idx+1:2d}: {sample}")
