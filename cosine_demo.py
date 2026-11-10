"""Hand-computed cosine similarity for defense/presentation."""
from src.pipeline import cosine_manual

# Các cặp cố định giúp minh họa các trường hợp cosine cơ bản khi trình bày.
pairs=[([1,0],[1,0]),([1,0],[0,1]),([1,1],[2,1]),([1,0],[-1,0])]
for a,b in pairs:
    print(f"A={a}, B={b}, cosine={cosine_manual(a,b):.4f}")
