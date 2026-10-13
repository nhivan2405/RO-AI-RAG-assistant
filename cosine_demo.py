"""Hand-computed cosine similarity for defense/presentation."""
from src.pipeline import cosine_manual
pairs=[([1,0],[1,0]),([1,0],[0,1]),([1,1],[2,1]),([1,0],[-1,0])]
for a,b in pairs:
    print(f"A={a}, B={b}, cosine={cosine_manual(a,b):.4f}")
