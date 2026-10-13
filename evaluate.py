"""Simple retrieval evaluation: source-level hit@k and MRR, NOT chunk ground-truth metrics.
For proper Recall@K/Precision@K, manually label relevant chunk IDs in a bigger dataset.
"""
import json,argparse
from src.pipeline import load_documents,make_chunks,RagIndex

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",default="data"); args=p.parse_args()
    qs=json.load(open("data/evaluation_questions.json",encoding="utf-8"))
    idx=RagIndex();idx.build(make_chunks(load_documents(args.data)))
    positives=[q for q in qs if q.get("source")]
    total_hit=0; mrr=0
    for q in qs:
        hits=idx.search(q["question"],5)
        ranks=[i for i,h in enumerate(hits,1) if h["source"]==q["source"]] if q.get("source") else []
        rank=min(ranks) if ranks else None
        if q.get("source"):
            total_hit+=bool(rank);mrr+=1/rank if rank else 0
        print(json.dumps({"query":q["question"],"expected_source":q.get("source"),"first_source_rank":rank,"top_sources":[h["source"] for h in hits]},ensure_ascii=False))
    print(f"Source Hit@5={total_hit/len(positives):.3f}; Source MRR={mrr/len(positives):.3f}; NOT full chunk relevance evaluation")
    print("Out-of-scope questions require abstention assessment at the LLM layer, not merely a source match.")
if __name__=="__main__":main()
