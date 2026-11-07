"""
The prewarm policy, the same for every predictor: which of the ranked actions become PrewarmRequests.

  theta  every tool with p >= theta
  topk   the top_k tools, each with p >= p_min (the floor keeps quiet on steps that are almost surely RESPOND)
END and RESPOND are never prewarmed (they are not functions).
"""

from pulse.predictor.contract import NO_PREWARM


def prewarm_set(ranked, theta):
    """Tools with probability >= theta (FR-5), highest first."""
    return [t for t, p in ranked if p >= theta and t not in NO_PREWARM]


def prewarm_select(ranked, policy="theta", theta=0.3, top_k=2, p_min=0.0):
    """Which tools to prewarm, from [(action, p), ...] highest first."""
    if policy == "theta":
        return prewarm_set(ranked, theta)
    if policy == "topk":
        return [t for t, p in [(t, p) for t, p in ranked if t not in NO_PREWARM][:top_k] if p >= p_min]
    raise ValueError(f"unknown PREWARM_POLICY {policy!r}; use theta or topk")
