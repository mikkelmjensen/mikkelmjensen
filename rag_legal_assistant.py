"""
RAG-based legal assistant for contract Q&A and clause extraction.
Retrieves relevant precedents and answers questions over legal document corpora.
"""

from dataclasses import dataclass
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


@dataclass
class LegalQueryResult:
    answer: str
    source_clauses: list[str]
    confidence: str


class LegalRAGAssistant:
    def __init__(self, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=80,
            separators=["

", "
", ".", " "]
        )
        self.vectorstore = None
        self.chain = None

    def ingest(self, documents: list[str]) -> int:
        chunks = self.splitter.create_documents(documents)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
        )
        return len(chunks)

    def query(self, question: str) -> LegalQueryResult:
        if not self.chain:
            raise RuntimeError("Ingest documents first.")
        result = self.chain({"query": question})
        sources = [doc.page_content[:200] for doc in result.get("source_documents", [])]
        return LegalQueryResult(
            answer=result["result"],
            source_clauses=sources,
            confidence="high" if len(sources) >= 3 else "medium",
        )
