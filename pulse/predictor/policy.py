"""
The prewarm policy, the same for every predictor: which of the ranked actions become PrewarmRequests.

  adaptive  (default) the top_k_max (3) most likely tools, each with p >= p_min, as CANDIDATES: each carries its
            rank and hit rate (calibration.py), and the controller warms only those worth a pod slot now
            (pulse/controller/adaptive.py: hit rate x the tool's cold start vs the slot's price). So how many tools a
            step warms -- the k of top-k -- follows the predictor's confidence and how full the cluster is.
  topk      exactly the top_k most likely tools, each with p >= p_min (a fixed k, for comparisons)
END and RESPOND are never prewarmed (they are not functions). The p_min floor keeps quiet on steps that are almost
surely a reply to the customer.
"""

from pulse.predictor.contract import NO_PREWARM

POLICIES = ("adaptive", "topk")


def prewarm_select(ranked, policy="adaptive", top_k=2, p_min=0.05, top_k_max=3):
    """Which tools to prewarm (topk) or offer as candidates (adaptive), from [(action, p), ...] highest first."""
    if policy not in POLICIES:
        raise ValueError(f"unknown PREWARM_POLICY {policy!r}; use adaptive or topk")
    n = top_k if policy == "topk" else top_k_max
    return [t for t, p in [(t, p) for t, p in ranked if t not in NO_PREWARM][:max(1, int(n))] if p >= p_min]
