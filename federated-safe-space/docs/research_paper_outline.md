# Research Paper Outline — LAG-PSA

**Target journal:** *Big Data* (Sage / Mary Ann Liebert)

## Working titles
1. "LAG-PSA: Privacy-Preserving Federated Aggregation for Content Moderation over Low-Bandwidth, Intermittently Connected Mobile Clients"
2. "Safe Space Without Surveillance: A Federated Learning Platform for Privacy-Preserving Social Media Moderation"

## Contributions
1. LAG-PSA — a domain-tailored aggregation protocol for moderation FL under intermittent connectivity and bandwidth constraints: asynchronous rounds with staleness bounds, top-k sparse quantized updates with residual carry-over, pairwise-masked secure aggregation, distributed differential privacy.
2. Privacy analysis — threat model (honest-but-curious aggregator, malicious clients), secure-aggregation construction, DP composition and per-round epsilon accounting.
3. Empirical study — convergence, communication cost, dropout robustness vs FedAvg / FedProx / centralized upper bound, with component ablations.
4. Open full-stack reference platform (this repository).

## Section plan
| § | Section | Content |
|---|---|---|
| 1 | Introduction | Harm-before-removal; privacy/regulatory risk of centralized moderation data; why generic FL fails on phones; contributions |
| 2 | Related Work | FedAvg/FedProx/SCAFFOLD/async FL; Bonawitz secure aggregation; DP in FL; top-k/quantization/residual compression; toxicity detection; on-device federated NLP; gap table |
| 3 | System & Threat Model | Architecture; actors and adversary capabilities; no raw-text egress guarantee |
| 4 | LAG-PSA Protocol | Client/server pseudocode; staleness semantics; residual carry-over correctness; masking construction; DP calibration; convergence intuition |
| 5 | Implementation | React Native client, FastAPI + Postgres, aggregation service, Docker; instrumentation |
| 6 | Evaluation | See plan below |
| 7 | Discussion & Limitations | Heterogeneity; on-device label quality; poisoning/norm bounding; long-horizon epsilon budget; phone compute |
| 8 | Ethics | Consent flow, opt-out, no raw-text egress, fairness checks |
| 9 | Conclusion | + artifact/reproducibility statement |

## Evaluation plan
- Datasets: Jigsaw Toxicity / Civil Comments; HateXplain for ablation cross-check.
- Partitioning: non-IID by conversation thread / user cluster; N = 10–100 simulated clients.
- Connectivity: random dropouts 0–60%; upload caps ~100 kbps–1 Mbps; mid-round disconnects.
- Baselines: centralized (upper bound), FedAvg, FedProx, naive async FedAvg.
- Metrics: per-round upload bytes; rounds-to-target-AUC; macro-F1/AUC on held-out test; participation; DP noise impact.
- Ablations: remove one of async window / top-k+quantization / residual carry-over / DP noise at a time.
- Key claim to evidence: maintained moderation AUC under 40% dropout where synchronous FedAvg stalls, at substantially fewer uploaded bytes (fill exact numbers from runs).

## Abstract skeleton
Social platforms detect harmful content with models trained on centrally collected user data, creating privacy risk and regulatory exposure — while harmful content harms users before removal. We present a full-stack platform in which a moderation classifier is trained federatively across mobile devices, so raw text never leaves the device. We contribute LAG-PSA, an aggregation protocol combining asynchronous staleness-bounded rounds, top-k quantized updates with residual carry-over, pairwise-masked secure aggregation, and distributed DP, tuned for low-bandwidth intermittently connected clients. Evaluation on public toxicity corpora partitioned non-IID under controlled dropout and bandwidth caps shows [RESULTS]. Reference implementation: React Native, FastAPI, PostgreSQL, Docker.

## Build/eval roadmap (~10 weeks)
W1–2 backend → W3–4 mobile client → W5–6 Flower integration + simulation → W7–8 protocol metrics + ablations → W9–10 evaluation runs + paper draft.
