import argparse
import os

from dotenv import load_dotenv

from src.chunking.chunker import chunk_documents
from src.config import load_config
from src.pipeline import RagIndex, answer_with_gemini, load_documents


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="Sinh viên được mượn bao nhiêu quyển sách?")
    parser.add_argument("--no-llm", action="store_true", help="retrieval only; no Gemini API")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    config = load_config()
    documents = load_documents(args.data)
    chunk_size = int(os.getenv("CHUNK_SIZE", config["chunking"]["chunk_size"]))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", config["chunking"]["chunk_overlap"]))
    chunks = chunk_documents(
        documents,
        chunk_size,
        chunk_overlap,
        strategy=config["chunking"]["strategy"],
    )
    index = RagIndex()
    dimension = index.build(chunks)
    top_k = int(os.getenv("TOP_K", config["retrieval"]["top_k"]))
    hits = index.search(args.question, top_k)
    print(f"Model={index.model_name}; dimension={dimension}; chunk count={len(chunks)}")
    for position, hit in enumerate(hits, start=1):
        print(
            f"[{position}] score={hit['score']:.4f} source={hit['source']} "
            f"page={hit['page']} | {hit['text'][:200]}"
        )
    if not args.no_llm:
        print("\nGEMINI ANSWER:\n", answer_with_gemini(args.question, hits))


if __name__ == "__main__":
    main()
