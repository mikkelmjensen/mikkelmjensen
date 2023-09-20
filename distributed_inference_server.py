"""
Distributed ML inference server with request batching and async queue.
High-throughput model serving for production legal AI pipelines.
"""

import asyncio
from dataclasses import dataclass
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class InferenceRequest:
    request_id: str
    text: str


@dataclass
class InferenceResult:
    request_id: str
    label: str
    score: float


class InferenceRequest(BaseModel):
    request_id: str
    text: str


class DistributedInferenceServer:
    def __init__(self, model_name: str, max_batch_size: int = 16, timeout_ms: int = 50):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue: asyncio.Queue = asyncio.Queue()
        self.app = FastAPI()
        self._register_routes()

    def _register_routes(self):
        @self.app.post("/infer")
        async def infer(req: InferenceRequest):
            future = asyncio.get_event_loop().create_future()
            await self.queue.put((req, future))
            return await future

    async def _process_batch(self, batch: list):
        requests, futures = zip(*batch)
        texts = [r.text[:512] for r in requests]
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        for i, future in enumerate(futures):
            label_idx = probs[i].argmax().item()
            future.set_result({"label": str(label_idx), "score": round(float(probs[i][label_idx]), 3)})

    async def run_batcher(self):
        while True:
            batch = []
            try:
                item = await asyncio.wait_for(self.queue.get(), timeout=self.timeout_ms / 1000)
                batch.append(item)
                while len(batch) < self.max_batch_size and not self.queue.empty():
                    batch.append(self.queue.get_nowait())
            except asyncio.TimeoutError:
                pass
            if batch:
                await self._process_batch(batch)
