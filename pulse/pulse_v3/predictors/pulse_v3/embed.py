"""
Sentence embeddings from the local llama.cpp embedding server (live/serve_embed.sh), cached on disk.

nomic-embed-text-v1.5 expects a task prefix; "classification: " is the one for features fed to a
classifier. Vectors are L2-normalised. The cache (data/embed_cache.npz) maps sha1(prefix + text) to the
vector, so every text is embedded once across runs.

Settings: PULSE_EMBED_BASE (default http://127.0.0.1:8091/v1).
"""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from ... import paths

BASE = os.getenv("PULSE_EMBED_BASE", "http://127.0.0.1:8091/v1")
CACHE = paths.DATA / "embed_cache.npz"
PREFIX = "classification: "
MAX_CHARS = 4000            # ~1000 tokens, well inside the model's 2048-token context
DIM = 768


class Embedder:
    def __init__(self, cache=CACHE, batch=64, timeout=300):
        self.cache_path, self.batch, self.timeout = Path(cache), batch, timeout
        self.cache = {}
        if self.cache_path.exists():
            z = np.load(self.cache_path)
            self.cache = dict(zip(z["keys"].tolist(), z["vecs"]))
        self.new = 0

    @staticmethod
    def key(text):
        return hashlib.sha1((PREFIX + text).encode()).hexdigest()

    def _request(self, texts):
        body = json.dumps({"model": "nomic-embed", "input": [PREFIX + t[:MAX_CHARS] for t in texts]}).encode()
        req = urllib.request.Request(BASE + "/embeddings", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())["data"]
        vecs = np.array([d["embedding"] for d in sorted(data, key=lambda d: d["index"])], dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)

    def __call__(self, texts):
        """(n, 768) array for a list of texts; only texts not in the cache go to the server."""
        keys = [self.key(t) for t in texts]
        todo = sorted({k: t for k, t in zip(keys, texts) if k not in self.cache}.items())
        for i in range(0, len(todo), self.batch):
            chunk = todo[i:i + self.batch]
            for (k, _), v in zip(chunk, self._request([t for _, t in chunk])):
                self.cache[k] = v
            self.new += len(chunk)
        return np.stack([self.cache[k] for k in keys]) if keys else np.zeros((0, DIM), np.float32)

    def save(self):
        if self.new:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            keys = list(self.cache)
            np.savez(self.cache_path, keys=np.array(keys), vecs=np.stack([self.cache[k] for k in keys]))
            self.new = 0
