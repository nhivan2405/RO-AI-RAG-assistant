import argparse
import os

from dotenv import load_dotenv

from src.chunking.chunker import chunk_documents
from src.config import load_config
from src.llm.rag_service import answer_with_rag
from src.pipeline import RagIndex, load_documents
from src.retrieval.retriever import FALLBACK_MESSAGE, Retriever
from src.utils.helpers import get_logger, setup_logging


def _print_retrieval_diagnostic(hits, retriever: Retriever, is_answerable: bool) -> None:
    """In trạng thái ngưỡng để hỗ trợ kiểm tra retrieval không dùng LLM."""
    top_score = f"{float(hits[0]['score']):.4f}" if hits else "N/A"
    decision = "ACCEPTED" if is_answerable else "REJECTED"
    print(
        f"Retrieval decision: {decision}; top_score={top_score}; "
        f"min_score={retriever.min_score:.4f}"
    )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="Sinh viên được mượn bao nhiêu quyển sách?")
    parser.add_argument("--no-llm", action="store_true", help="retrieval only; no Gemini API")
    parser.add_argument("--gemini", action="store_true", help="call Gemini with grounded evidence")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    config = load_config()
    setup_logging(config)
    logger = get_logger("cli")

    try:
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
        retriever = Retriever(config["retrieval"]["min_score"])
        is_answerable = retriever.is_answerable(hits)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        logger.error("application_failed error_type=%s", type(error).__name__)
        print(f"ERROR: {error}")
        return

    print(f"Model={index.model_name}; dimension={dimension}; chunk count={len(chunks)}")
    for position, hit in enumerate(hits, start=1):
        print(
            f"[{position}] score={hit['score']:.4f} source={hit['source']} "
            f"page={hit['page']} | {hit['text'][:200]}"
        )
    if args.no_llm or not args.gemini:
        _print_retrieval_diagnostic(hits, retriever, is_answerable)
        if not is_answerable:
            print("\nRETRIEVAL FALLBACK:\n", FALLBACK_MESSAGE)
        return

    if is_answerable:
        try:
            result = answer_with_rag(
                args.question,
                index,
                top_k=top_k,
                min_score=config["retrieval"]["min_score"],
                token_budget=config["prompts"]["context_token_budget"],
                model_name=os.getenv("GEMINI_MODEL", config["models"]["gemini"]),
                hits=hits,
                timeout_seconds=config["llm"]["timeout_seconds"],
            )
        except RuntimeError as error:
            logger.error("gemini_answer_failed error_type=%s", type(error).__name__)
            print(f"ERROR: {error}")
            return
        print("\nGEMINI ANSWER:\n", result.answer)
    else:
        print("\nGEMINI ANSWER:\n", FALLBACK_MESSAGE)


if __name__ == "__main__":
    main()
