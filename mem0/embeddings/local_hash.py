import hashlib
import math
import re
from typing import Literal, Optional

from mem0.embeddings.base import EmbeddingBase


_TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


class LocalHashEmbedding(EmbeddingBase):
    """Dependency-free local hashing embedder.

    This is intended for private/local deployments that need lexical retrieval
    without sending text to an external embedding API. It is deterministic and
    not a replacement for a trained semantic embedding model, but it is good
    enough for smoke tests and simple keyword-style memory lookup.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.config.embedding_dims = self.config.embedding_dims or 1536

    def embed(self, text, memory_action: Optional[Literal["add", "search", "update"]] = None):
        dims = int(self.config.embedding_dims)
        vector = [0.0] * dims
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector
