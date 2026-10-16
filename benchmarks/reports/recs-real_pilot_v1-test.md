# Held-out recommendation check: real_pilot@v1 test (held out)

Commit `cab6bd2fe04743a0360ca6760523e7ed044a9b86`; manifest `19039096dcc201ac…`; quality features: heuristic baseline (quality features are heuristic, not AI or human).
> **Labels:** topical proxy: persona's matching publisher show; no human relevance or quality labels. 5 personas × 17 items: indicative only, no confidence claimed.

| embedder | nDCG@5 | P@5 | R@5 |
|---|---:|---:|---:|
| hashing | 0.7868 | 0.4 | 0.7238 |
| minilm_onnx | 0.6877 | 0.4 | 0.7238 |

| embedder | persona | relevant | nDCG | shows in top-5 |
|---|---|---:|---:|---|
| hashing | frontend_dev | 3 | 0.765 | {'changelog:jsparty': 2, 'talkpython': 1} |
| hashing | sre | 2 | 0.85 | {'changelog:shipit': 2, 'talkpython': 2, 'changelog:jsparty': 1} |
| hashing | pythonista | 7 | 0.553 | {'talkpython': 2, 'changelog:brainscience': 1, 'changelog:jsparty': 1} |
| hashing | mind_curious | 2 | 1.0 | {'changelog:brainscience': 2, 'talkpython': 1} |
| hashing | oss_generalist | 3 | 0.765 | {'changelog:podcast': 2, 'talkpython': 1} |
| minilm_onnx | frontend_dev | 3 | 0.704 | {'changelog:jsparty': 2, 'talkpython': 2, 'changelog:shipit': 1} |
| minilm_onnx | sre | 2 | 0.651 | {'talkpython': 2, 'changelog:shipit': 2, 'changelog:jsparty': 1} |
| minilm_onnx | pythonista | 7 | 0.553 | {'talkpython': 2, 'changelog:jsparty': 1} |
| minilm_onnx | mind_curious | 2 | 1.0 | {'changelog:brainscience': 2, 'talkpython': 1} |
| minilm_onnx | oss_generalist | 3 | 0.531 | {'talkpython': 2, 'changelog:podcast': 2, 'changelog:jsparty': 1} |
