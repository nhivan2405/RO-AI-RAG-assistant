"""Character-based chunk sweep; diagnostics only, not proof of retrieval accuracy."""
import csv
from src.pipeline import load_documents,make_chunks
rows=[]; docs=load_documents("data")
for size in (300,700,1100):
    for overlap in (0,100):
        if overlap>=size:continue
        chunks=make_chunks(docs,size,overlap)
        lens=[len(c.text) for c in chunks]
        rows.append({"chunk_size_chars":size,"overlap_chars":overlap,"num_chunks":len(chunks),"avg_chunk_length_chars":round(sum(lens)/len(lens),1) if lens else 0,"max_chunk_length_chars":max(lens) if lens else 0})
with open("results/chunk_diagnostics.csv","w",newline="",encoding="utf-8-sig") as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
print(*rows,sep="\n")
