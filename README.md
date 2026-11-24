# FloodLens

AI-powered flood intelligence platform (demonstration build). See the danger before it spreads.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

- `app.py` - Streamlit interface (6 pages)
- `data.py` - illustrative Dubai demo reports and history (clearly labelled as demo data)
- `vision.py` - FloodLens Vision, the image analyser used by AI Image Analysis
- `.streamlit/config.toml` - theme

## Enable the AI engine

1. Create an API key at console.anthropic.com (needs a small credit balance).
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and paste your key in it.
3. Restart the app. The AI Image Analysis page then says "Photos are assessed by a vision AI model".

Without a key, or if the AI call fails, the app falls back to the local prototype analyser.

## Notes

- All reports and history are sample data. Reports you submit live only in your browser session.
- Without an API key the image analyser is a colour- and texture-based prototype, not a trained neural network.
