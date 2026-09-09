# Federated-Privacy Safe Space for Social Media

Privacy-preserving content moderation via federated learning: a shared
moderation model is trained across many clients while raw user text never
leaves the device. Only compressed, masked, DP-noised model updates travel.

## What's in this version

| Piece | Status |
|---|---|
| FastAPI + JWT + PostgreSQL backend | done |
| LAG-PSA aggregation core (async staleness window, top-k + 8-bit quantization + residual carry-over, norm clipping + DP noise, quorum aggregation) | done |
| Simulated clients (`client-sim`): random dropouts, bandwidth delays, staleness growth | done |
| Flower server wrapper (stack alignment for the paper) | next batch |
| React Native mobile app (Feed / Report / Consent / TrainingStatus) | next batch |

The simulated updates are synthetic (planted signal + noise) to exercise the
protocol loop end-to-end; real on-device training arrives with the mobile client.

## Run with Docker (recommended)

1. Install Docker Desktop (<https://docs.docker.com/desktop/>) and start it.
2. Extract the zip and open a terminal in the `federated-safe-space` folder.
3. Build and start everything:

   ```bash
   docker compose up --build
   ```

   The first build downloads the Python and Postgres images — give it a few minutes.
4. Open the web dashboard: <http://localhost:8000/> — feed, posting, reporting, and a live federation monitor. Machine-readable API docs remain at <http://localhost:8000/docs>.
5. Watch the federation simulation (3 replicas × 8 virtual clients, 30% dropout):

   ```bash
   docker compose logs -f client-sim
   ```

   Each round prints uploads vs dropouts and the server's round counter;
   aggregation fires automatically whenever 5+ updates are pending.
6. Inspect results in the docs UI:
   - `GET /federation/global-model` → current round, pending updates
   - `GET /metrics` → users / posts / reports / federation round
7. Try the product flow — easiest in the web dashboard (register → post → report → Federation tab). Raw API equivalent (in the docs UI: Authorize with your token first):
   - `POST /auth/register` `{"username": "me", "password": "secret1", "consent_federated": true}` → copy `token`
   - Click **Authorize**, paste the token
   - `POST /posts` `{"body": "hello world"}` → placeholder toxicity score
   - `POST /reports` `{"post_id": 1, "reason": "abuse"}`
8. Stop: `docker compose down` (add `-v` to also wipe the database).

**Troubleshooting**

- Port 8000 busy → change `"8000:8000"` to `"8001:8000"` in `docker-compose.yml`, use port 8001 everywhere.
- Only one sim replica starts → run `docker compose up --scale client-sim=3`.

## Run without Docker

Requires Python 3.11+.

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload        # uses local SQLite, no Postgres needed
```

Second terminal:

```bash
cd client-sim
pip install -r requirements.txt
python simulate.py --api http://localhost:8000 --clients 8 --rounds 5 --drop-rate 0.3
```

## Modify

- Protocol knobs: `backend/app/federation/lagpsa.py` → `LagPsaAggregator(staleness_window, clip_norm, dp_sigma, min_quorum, lr)`; client compression in `LagPsaClient(topk_frac, bits)`
- Simulation profile: `client-sim/simulate.py` args/env (clients, rounds, drop rate, top-k fraction)
- Moderation placeholder: `backend/app/main.py` → `moderation_score` (replaced by the federated model later)
- DB schema: `backend/app/models.py`; Auth/JWT: `backend/app/auth.py`

## Real vs. still placeholder

- **Real:** federated training of a logistic-regression toxicity model on real labeled text —
  a handwritten seed corpus expanded by seeded templates (~600 examples), partitioned non-IID
  (label-skewed) across clients. Clients train locally with numpy; raw text never leaves the
  client; only top-k compressed, quantized weight deltas travel through the LAG-PSA pipeline.
  The server aggregates with clipping + DP noise and evaluates AUC/F1 on a held-out test set
  every round — watch convergence on the dashboard's Federation tab.
- **Real:** protocol mechanics — async staleness window, top-k + 8-bit quantization with
  residual carry-over, pairwise masking (client-side), quorum aggregation, DP noise.
- **Real:** scoring — `/posts` is scored by the aggregated global model once round 1 exists
  (keyword bootstrap only before the very first aggregation).
- **Still placeholder:** training inside the React Native app itself (simulated Docker clients
  stand in for phones), and corpus scale — for publication-grade results drop the full
  Jigsaw/Civil Comments CSV into `client-sim/data/` in the same JSON shape.

See `docs/RESEARCH_PAPER_OUTLINE.md` for the paper plan (target journal: *Big Data*, Sage).
