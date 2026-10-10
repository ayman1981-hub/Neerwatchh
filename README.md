# 💧 NEERWATCH — Street-level Early Warning for Unsafe Drinking Water

A portable, fully local early-warning prototype: citizen reports (Tamil/English)
flow through **three ML models** and a rule engine into an officer-facing
dashboard that turns a ward red when an outbreak signal crosses threshold.

- **ML #1 — Symptom extraction:** sentence-transformers embeddings + Logistic Regression
  (trained on `data/labelled_reports.csv`, 134 labelled messages) + regex ward/case counts
- **ML #2 — Clustering:** DBSCAN on 48h report embeddings, per ward
- **ML #3 — Anomaly:** Isolation Forest on simulated ward-day baselines
- **Rule:** `anomaly score > 0.70 AND ≥ 3 reports in 48h` → alert with reasons + actions
- **Rain:** Open-Meteo API with automatic fallback to `data/rain_cache.csv`
- **Storage:** local SQLite (`data/neerwatch.db`), no paid services anywhere

## Quick start (portable)

```bash
pip install -r requirements.txt
python run.py --replay
```

That one command:

1. initialises the SQLite database (17 wards, mock PWD leak logs),
2. trains/caches the two models into `models/` (first run also downloads the
   multilingual sentence-transformer once, ~470 MB — afterwards it is fully offline),
3. launches the Streamlit dashboard in a subprocess and prints its URL,
4. replays `data/demo_reports.csv` row by row with a 2–3 s delay, printing the
   pipeline stage for every report:

```
[extract] ward=9 symptom=diarrhoea
[cluster] ward 9 cluster size=4
[anomaly] score=0.85
[ALERT] Ward 9 crossed threshold. Triggering dashboard update.
```

The dashboard (3-second auto-refresh) turns **Ward 9 red** and shows the alert card
with reasons ("High Rain … + Cluster of 5 Similar Reports + …") and suggested actions.

> **Tip:** if port 8501 is busy, run.py picks the next free port and prints the URL.

## Commands

| Command | What it does |
|---|---|
| `python run.py` | live random simulation loop + dashboard |
| `python run.py --replay` | **scripted Ward 9 outbreak demo (use this for presentations)** |
| `python run.py --scenario quiet` | quiet scenario — proves no false alerts fire |
| `python run.py --scenario outbreak` | generated outbreak variant |
| `python run.py --no-dashboard` | terminal-only (CI / debugging) |
| `python run.py --delay 1 --port 8505 --keep-db` | tune pacing / port / keep history |
| `NEERWATCH_OFFLINE=1 python run.py --replay` | never touch the network (rain cache + model cache) |

Dashboard alone: `streamlit run app.py`.

### Live intake (demo bonus)

The dashboard has a **Live citizen report intake** box — type a message exactly as a
resident would, e.g.

```
வார்டு 5 தண்ணீரில் நாற்றம், 3 பேருக்கு வயிற்றுப்போக்கு
```

and it runs through the full pipeline; the map, table and feed update within 3 seconds.

## Real vs simulated (say it on screen — the dashboard states this too)

| Real | Simulated |
|---|---|
| Open-Meteo rainfall (with offline cache fallback) | citizen reports & the outbreak |
| The 17-ward list / geometry | PWD leak log |
| The trained models (LR, DBSCAN, Isolation Forest) | scenario generation in `src/simulate.py` |

**Out of scope (future):** Telegram / LLM / voice intake, resident alerts, login, storm mode.

## Evaluation

```bash
jupyter lab notebooks/evaluate.ipynb   # or: jupyter nbconvert --to notebook --execute notebooks/evaluate.ipynb
```

`notebooks/evaluate.ipynb` (pre-executed, charts included) measures on **simulated** data:

- **Lead time:** alert fires on report 3 of the 5-report outbreak → **60 min** after the
  first outbreak report (30-min arrival spacing),
- **False alarms:** **2 / 6,120 quiet ward-days = 0.01 per ward-month** across 204
  simulated ward-months.

Quote these as *behaviour on simulation*, never as real-world accuracy.

## Project structure

```
neerwatch/
├── run.py                  # entry point: dashboard + terminal loop / --replay
├── app.py                  # Streamlit dashboard (auto-refresh every 3s)
├── requirements.txt
├── .streamlit/config.toml  # bright, classic-formal light theme
├── data/                   # wards, leaks, demo + labelled reports, rain cache, SQLite db
├── src/                    # db, extract, cluster, rain, features, anomaly, alert, pipeline, simulate
├── models/                 # symptom_clf.pkl, iso_forest.pkl (auto-generated)
└── notebooks/evaluate.ipynb
```

## Troubleshooting

- **First run is slow** — the sentence-transformer downloads once; subsequent runs start in seconds. If the download fails, the built-in deterministic mock embedder keeps everything runnable.
- **No rain / weird rain** — Open-Meteo is best-effort; `data/rain_cache.csv` always answers, and `NEERWATCH_OFFLINE=1` skips the API entirely.
- **Port already in use** — run.py auto-selects the next free port; or pass `--port`.
- **Fresh demo state** — every run resets reports/snapshots/alerts by default; `--keep-db` preserves history.
