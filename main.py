import os,argparse
from dotenv import load_dotenv
from src.pipeline import load_documents,make_chunks,RagIndex,answer_with_gemini,cosine_manual

def main():
    load_dotenv()
    parser=argparse.ArgumentParser()
    parser.add_argument("--question",default="Sinh viên được mượn bao nhiêu quyển sách?")
    parser.add_argument("--no-llm",action="store_true",help="retrieval only; no Gemini API")
    parser.add_argument("--data",default="data")
    args=parser.parse_args()
    docs=load_documents(args.data)
    chunks=make_chunks(docs,int(os.getenv("CHUNK_SIZE","700")),int(os.getenv("CHUNK_OVERLAP","100")))
    idx=RagIndex()
    dimension=idx.build(chunks)
    hits=idx.search(args.question,int(os.getenv("TOP_K","5")))
    print(f"Model={idx.model_name}; dimension={dimension}; chunk count={len(chunks)}")
    for i,h in enumerate(hits,1):
        print(f"[{i}] score={h['score']:.4f} source={h['source']} page={h['page']} | {h['text'][:200]}")
    if not args.no_llm:
        print("\nGEMINI ANSWER:\n",answer_with_gemini(args.question,hits))
if __name__=="__main__": main()
