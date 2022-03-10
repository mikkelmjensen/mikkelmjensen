"""
Legal document classifier using transformer-based NLP.
Classifies contracts, briefs, judgments, and regulatory filings for AI-powered legal workflows.
"""

from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


DOC_TYPES = ["contract", "brief", "judgment", "regulation", "filing", "correspondence"]


@dataclass
class ClassificationResult:
    doc_type: str
    confidence: float
    top_labels: list
    excerpt: str


class LegalDocumentClassifier:
    def __init__(self, model_name: str = "nlpaueb/legal-bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.doc_types = DOC_TYPES

    def classify(self, text: str) -> ClassificationResult:
        inputs = self.tokenizer(text[:512], return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        top_idx = probs.argsort(descending=True)[:3].tolist()
        label_idx = top_idx[0]
        doc_type = self.doc_types[label_idx % len(self.doc_types)]
        top_labels = [
            {"label": self.doc_types[i % len(self.doc_types)], "score": round(float(probs[i]), 3)}
            for i in top_idx
        ]
        return ClassificationResult(
            doc_type=doc_type,
            confidence=round(float(probs[label_idx]), 3),
            top_labels=top_labels,
            excerpt=text[:120],
        )

    def batch_classify(self, texts: list) -> list:
        return [self.classify(t) for t in texts]
