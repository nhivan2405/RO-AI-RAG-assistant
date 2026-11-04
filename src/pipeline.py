"""Điều phối RAG demo với embedding và Qdrant local persistent."""
import os
import numpy as np
from dotenv import load_dotenv

from src.chunking.chunker import Chunk, make_chunks
from src.config import load_config
from src.embeddings.embedder import Embedder
from src.ingestion.loader import load_documents
from src.vectordb.vector_store import PersistentVectorStore

load_dotenv()

def cosine_manual(a, b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    if den == 0: raise ValueError("Zero vector")
    return float(np.dot(a,b)/den)


class RagIndex:
    """Tương thích ngược cho index RAG dùng persistent vector store."""

    def __init__(self, model_name=None, collection=None):
        config = load_config()
        self.model_name = model_name or os.getenv(
            "EMBED_MODEL", config["models"]["embedding"]
        )
        self.embedder = Embedder(self.model_name)
        self.collection = collection or config["vectordb"]["collection_name"]
        self.vector_store = PersistentVectorStore(
            storage_path=config["vectordb"]["storage_path"],
            collection_name=self.collection,
            model_name=self.model_name,
            vector_dimension=self.embedder.expected_dimension,
        )
        self.client = self.vector_store.client
        self.chunks = []

    def build(self, chunks):
        if not chunks:
            raise ValueError("No usable chunks")
        self.chunks = chunks
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        dimension = self.embedder.dimension
        if dimension != self.vector_store.vector_dimension:
            raise ValueError("Embedding dimension does not match vector store dimension")
        self.vector_store.index_chunks(chunks, vectors)
        return dimension

    def search(self, question, top_k=5):
        query = self.embedder.embed_query(question)
        return self.vector_store.search(query, top_k)

def make_prompt(question,hits):
    context="\n\n".join(f"[{i}] Nguồn: {h['source']}, trang: {h['page'] or 'N/A'}\n{h['text']}" for i,h in enumerate(hits,1))
    return f"""Bạn là trợ lý hỏi đáp tài liệu. Chỉ dùng CONTEXT dưới đây.
Nếu không đủ dữ liệu, hãy trả lời: 'Không tìm thấy đủ thông tin trong tài liệu được cung cấp.'
Không bịa dữ kiện hay nguồn. Ghi nguồn như [1], [2] khi dùng thông tin.
CONTEXT:\n{context}\n\nCÂU HỎI: {question}\nTRẢ LỜI:"""

def answer_with_gemini(question,hits):
    key=os.getenv("GEMINI_API_KEY")
    if not key or key=="YOUR_API_KEY_HERE":
        raise RuntimeError("Set GEMINI_API_KEY in .env; use --no-llm to test retrieval without an API key")
    from google import genai
    client=genai.Client(api_key=key)
    model_name = os.getenv("GEMINI_MODEL", load_config()["models"]["gemini"])
    resp=client.models.generate_content(model=model_name,contents=make_prompt(question,hits))
    return resp.text or "(API returned no text)"
