# Rank One Radar — POC

UNT College of Information proof-of-concept on **synthetic data** that replicates the
Rank One Partner API v1.0 schema (April 2026). No real athlete data anywhere.

## Views
- **Radar Insights** — leadership view: athlete-days lost, a recovery-prediction model
  benchmarked against trainers' own return estimates, the premature-return re-injury
  multiplier, and a ranked watchlist of open cases.
- **Athlete Focus** — one athlete's full history: injury timeline, pain trajectories,
  recovery vs peers, case files with AI summarization.
- **Injury Focus** — population analytics: season shape, recovery distributions,
  estimated-vs-actual accuracy, concussion protocol compliance, AI briefing.

## Run locally
```
pip install -r requirements.txt
python generate_data.py    # regenerates data/ (already included)
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repo (keep it **private** — the schema layout mirrors
   a confidential partner document).
2. At https://share.streamlit.io → New app → pick the repo, branch `main`, file `app.py`.
3. In the app's **Settings → Secrets**, paste (values from your local `.env`):
   ```toml
   VITE_AOAI_ENDPOINT = "..."
   VITE_AOAI_KEY = "..."
   VITE_AOAI_DEPLOYMENT = "..."
   VITE_AOAI_API_VERSION = "..."
   ```
   Without these the app still runs; the AI buttons show as offline.

`.env` files are git-ignored — never commit credentials.
