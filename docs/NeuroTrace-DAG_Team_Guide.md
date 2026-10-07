# NeuroTrace-DAG — viva card (reliability framing)

## One sentence

On Junyi, next-item Recall@K is mostly **continue** (same exercise again, ~80%). We report that Inertia Illusion, the nulls, and move the DAG to **correctness / readiness** — we do **not** claim a recommender that beats last-item.

## Numbers

| Claim | Number |
| --- | --- |
| continue / revisit / advance | ~0.80 / 0.09 / 0.11 |
| RAGR vs last-item (review) | 0.891 vs **0.893** (we lose) |
| Advance RAGR vs sim | 0.065 vs **0.102** (we lose) |
| Memory | logistic 0.864 [0.861, 0.867] on all 449,453 revisit rows (0.877 on first 80k is superseded) > HLR 0.789 |
| Timed forgetting | Δ −0.0029 → null |
| DAG pedagogy | unlocked 0.769 vs violate 0.668 |

## Attacks

1. **You lose to last-item.** Yes on the easy mass — that’s the finding. Last-item never advances.  
2. **RepeatNet did the gate.** Cite it; we claim educational validity analysis, not mechanism novelty.  
3. **Forgetting?** Test on **revisit** + gap bins, not on continue.  
4. **DAG ablation null.** Right for ranking; wrong target — use readiness→correctness.  
5. **OULAD?** No shared concept id.

## Do not say

“Phase 5 is the novel module.” “We beat baselines.” “Embeddings find prerequisites.”
