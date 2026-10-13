"""Demo RAG: document -> chunk -> embed -> local Qdrant -> search -> Gemini.
No training/fine-tuning takes place; embeddings are calculated by a pretrained encoder.
"""
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os
import uuid
import numpy as np
from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

load_dotenv()

@dataclass
class Chunk:
    text: str
    source: str
    page: int | None
    chunk_id: str

def load_documents(folder: str = "data") -> list[tuple[str, int | None, str]]:
    docs=[]
    for path in sorted(Path(folder).glob("*")):
        if path.suffix.lower()==".txt":
            docs.append((path.name,None,path.read_text(encoding="utf-8")))
        elif path.suffix.lower()==".pdf":
            for index,page in enumerate(PdfReader(str(path)).pages,1):
                value=page.extract_text() or ""
                if value.strip(): docs.append((path.name,index,value))
        elif path.suffix.lower()==".docx":
            doc=Document(path)
            value="\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if value.strip(): docs.append((path.name,None,value))
    return docs

def make_chunks(docs, chunk_size=700, chunk_overlap=100):
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require 0 <= chunk_overlap < chunk_size")
    splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n","\n",". ","; "," ",""],
        chunk_size=chunk_size,chunk_overlap=chunk_overlap,
        length_function=len,  # IMPORTANT: CHARACTER counts, not tokenizer tokens
    )
    chunks=[]
    for source,page,text in docs:
        for i,part in enumerate(splitter.split_text(text)):
            if part.strip():
                chunks.append(Chunk(part,source,page,f"{source}:p{page}:c{i}"))
    return chunks

def cosine_manual(a,b):
    a=np.asarray(a,dtype=float); b=np.asarray(b,dtype=float)
    den=np.linalg.norm(a)*np.linalg.norm(b)
    if den == 0: raise ValueError("Zero vector")
    return float(np.dot(a,b)/den)

class RagIndex:
    def __init__(self, model_name=None, collection="rag_demo"):
        self.model_name=model_name or os.getenv("EMBED_MODEL","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.embedder=SentenceTransformer(self.model_name)
        self.client=QdrantClient(":memory:")  # local prototype; data lost when process exits
        self.collection=collection
        self.chunks=[]

    def build(self,chunks):
        if not chunks: raise ValueError("No usable chunks")
        self.chunks=chunks
        vecs=self.embedder.encode([c.text for c in chunks],normalize_embeddings=True,show_progress_bar=False)
        dimension=int(vecs.shape[1])  # 384 for MiniLM, 1024 for BGE-M3
        self.client.create_collection(self.collection,vectors_config=models.VectorParams(size=dimension,distance=models.Distance.COSINE))
        records=[]
        for c,v in zip(chunks,vecs):
            records.append(models.PointStruct(id=str(uuid.uuid4()),vector=v.tolist(),payload={"text":c.text,"source":c.source,"page":c.page,"chunk_id":c.chunk_id}))
        self.client.upsert(collection_name=self.collection,points=records)
        return dimension

    def search(self,question,top_k=5):
        query=self.embedder.encode([question],normalize_embeddings=True,show_progress_bar=False)[0]
        result=self.client.query_points(collection_name=self.collection,query=query.tolist(),limit=top_k,with_payload=True)
        return [{"score":float(hit.score), **hit.payload} for hit in result.points]

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
    resp=client.models.generate_content(model=os.getenv("GEMINI_MODEL","gemini-2.5-flash"),contents=make_prompt(question,hits))
    return resp.text or "(API returned no text)"
