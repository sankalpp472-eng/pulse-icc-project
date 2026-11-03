"""
pulse-v3: next-tool predictor from the conversation text, the tool graph and the environment state.

  predictor.py  PulseV3Predictor: one session's hints -> Predictions; bundle loading
  models.py     Semantic (V3), MultiTool, SelfHint, Markov baseline, Oracle; Ctx, contexts, metrics
  graph.py      schema-derived tool graph and per-hint ready-set features
  state.py      environment-state flags from user messages and tool results
  embed.py      sentence embeddings from the local llama.cpp server
  online.py     per-agent-model online adapter on top of the frozen model

Served by pulse_pred/service.py. Training code and experiments live in the original pulse repository.
"""
