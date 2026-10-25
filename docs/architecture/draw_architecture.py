"""
Pulse architecture diagram for slides (16:9).

    py docs/architecture/draw_architecture.py
writes pulse_architecture.png (200 dpi) and pulse_architecture.svg next to this script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent

INK, MUTED, LINE = "#1f2328", "#57606a", "#8c959f"
CLIENT_BG, CLOUD_BG = "#f3f6fa", "#f2f8f5"
AGENT, PRED, CTRL, ING, KN, DB, API = (
    "#dbe9fb", "#e4dcfa", "#fde2cf", "#fff1c2", "#d5efe4", "#fbd9d9", "#e6e8eb")
FLOW_CALL, FLOW_PRED = "#2a78d6", "#1baf7a"   # tool-call path (blue), prewarm path (green)
DASH = (0, (4, 2))

fig = plt.figure(figsize=(13.333, 7.5))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 56.25)
ax.axis("off")


def box(x, y, w, h, color, title, body="", edge=LINE, title_size=11.5, body_size=8.8, lw=1.2, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=color, ec=edge, lw=lw))
    if body:
        ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 3.6, body, ha="center", va="top", fontsize=body_size,
                color=MUTED, linespacing=1.35)
    elif title:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=INK)


def label(x, y, text, color, ha="center", size=8.8):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size, color=color, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=1.2, alpha=0.95))


def arrow(p, q, color, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, lw=1.8, color=color,
                                 linestyle=ls, shrinkA=2, shrinkB=2))


# ---------------------------------------------------------------- title
ax.text(2, 54.2, "Pulse: predictive prewarming of serverless MCP tools for LLM agents",
        fontsize=17, fontweight="bold", color=INK, va="center")
ax.text(2, 51.6, "Predict the agent's next tool call and wake its pod before the call arrives, "
                 "hiding the cold start", fontsize=10.5, color=MUTED, va="center")

# ---------------------------------------------------------------- tiers
box(1.5, 6.5, 37, 42.5, CLIENT_BG, "", edge="#c9d3df", lw=1.4, r=1.2)
ax.text(3, 47.4, "CLIENT TIER  ·  laptop", fontsize=10.5, fontweight="bold", color="#3b5b84", va="center")
box(41.5, 6.5, 57, 42.5, CLOUD_BG, "", edge="#bcd9cb", lw=1.4, r=1.2)
ax.text(43, 47.4, "CLOUD TIER  ·  Kubernetes (local kind / AWS EKS, 2 workers)", fontsize=10.5,
        fontweight="bold", color="#2f7356", va="center")

# ---------------------------------------------------------------- client tier
box(4, 39.2, 32, 6.3, "#ffffff", "Simulated customer",
    "LLM role-plays the τ-bench airline customer")
box(4, 26.8, 32, 9.6, AGENT, "ReAct agent",
    "reason → act → observe loop  ·  LLM: Groq gpt-oss-120b\n"
    "or trace replayer: τ-bench traces, fixed seed 252\n"
    "sends a prediction hint before every tool call")
box(4, 15.2, 32, 8.8, PRED, "Prediction service",
    "FastAPI + scikit-learn\n"
    "ensemble: workflow graph + history + trajectory\n"
    "ranked next-tool probabilities  ·  threshold θ = 0.3")
box(4, 8.2, 15.5, 5.6, "#ffffff", "Measurement", "pod watcher + call log\ncalls.csv · pods.csv",
    title_size=10, body_size=8)
box(20.5, 8.2, 15.5, 5.6, CTRL, "Prewarm controller", "watches CRD, sends\nwarm-up MCP ping",
    title_size=10, body_size=8)

# ---------------------------------------------------------------- cloud tier
box(44, 23, 11, 16.5, ING, "Kourier",
    "ingress /\nload balancer\n\nroutes each call\nby Host header\nto its tool\nservice",
    title_size=10.5, body_size=8)
box(58, 40.4, 38, 5.9, KN, "Knative Serving: activator + autoscaler",
    "scales each tool 0 ↔ 1 pod  ·  idle window 60 s = TTL\n"
    "min-scale 0 (C0, C3) or 1 (C1)  ·  holds calls while a pod starts",
    title_size=10.5, body_size=8.2)

# 14 tool services
tx, ty, tw, th = 58, 21.0, 38, 18.6
box(tx, ty, tw, th, "#ffffff", "", r=1.0)
ax.text(tx + tw / 2, ty + th - 1.3, "14 MCP tool services (one Knative service + pod each)",
        ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
tools = ["get_user_details", "get_reservation_details", "search_direct_flight", "search_onestop_flight",
         "list_all_airports", "calculate", "think",
         "book_reservation", "cancel_reservation", "update_reservation_flights",
         "update_reservation_passengers", "update_reservation_baggages", "send_certificate",
         "transfer_to_human_agents"]
for i, t in enumerate(tools):
    col, row = divmod(i, 7)
    x0, y0 = tx + 1.2 + col * 18.4, ty + th - 5.0 - row * 1.9
    ax.add_patch(FancyBboxPatch((x0, y0), 17.2, 1.6, boxstyle="round,pad=0,rounding_size=0.4",
                                fc="#fdecea" if i >= 7 else "#eaf3fd", ec="none"))
    ax.text(x0 + 8.6, y0 + 0.8, t, ha="center", va="center", fontsize=7.6, color=INK, family="monospace")
ax.text(tx + 1.2, ty + 0.9, "■", color="#bfd9f7", fontsize=9, va="center")
ax.text(tx + 2.6, ty + 0.9, "read tools", color=MUTED, fontsize=8, va="center")
ax.text(tx + 12, ty + 0.9, "■", color="#f6c8c2", fontsize=9, va="center")
ax.text(tx + 13.4, ty + 0.9, "write tools", color=MUTED, fontsize=8, va="center")
ax.text(tx + tw - 1.2, ty + 0.9, "MCP Python SDK · Streamable HTTP", color=MUTED, fontsize=8,
        va="center", ha="right")

box(44, 8.2, 25, 10.8, API, "Kubernetes API server",
    "PrewarmRequest CRD (one object per tool)\n"
    "spec ← predictor: tool, P, θ, requestedAt\n"
    "status ← controller: Warming → Ready → Evicted",
    title_size=10, body_size=8)
box(72, 8.2, 24, 10.8, DB, "Redis (shared DB)", "users · reservations · flights\nreset to seed copy per run",
    title_size=10, body_size=8)

# ---------------------------------------------------------------- flows
# conversation
arrow((20, 39.2), (20, 36.4), INK, style="<|-|>")
ax.text(21, 37.8, "conversation", fontsize=8.5, color=MUTED, va="center")
# 1 hint
arrow((14, 26.8), (14, 24.0), FLOW_PRED)
label(15.2, 25.4, "① hint: workflow, trajectory", FLOW_PRED, ha="left")
# 2 tool call: agent -> ingress -> tool; result back
arrow((36, 32.5), (44, 32.5), FLOW_CALL)
label(40, 35.1, "② tool call\n(MCP / HTTP)", FLOW_CALL)
arrow((55, 32.5), (58, 32.5), FLOW_CALL)
arrow((44, 29.5), (36, 29.5), FLOW_CALL)
label(40, 27.6, "result +\nlatency", FLOW_CALL, size=8.2)
# 3 predictor writes the PrewarmRequest
arrow((36, 17.0), (44, 17.0), FLOW_PRED)
label(39.3, 19.4, "③ P ≥ θ:\nrequest", FLOW_PRED, size=8.2)
# 4 controller watches the CRD
arrow((44, 10.0), (36, 10.0), FLOW_PRED)
label(40, 8.4, "④ watch", FLOW_PRED, size=8.2)
# 5 warm-up ping through the ingress (same path as a real call)
ax.plot([36, 42.6, 42.6], [12.6, 12.6, 24.6], color=FLOW_PRED, lw=1.8, ls=DASH)
arrow((42.6, 24.6), (44, 24.6), FLOW_PRED, ls=DASH)
label(39.3, 14.3, "⑤ warm-up\nping", FLOW_PRED, size=8.2)
# knative scales tools; tools <-> redis
ax.plot([77, 77], [40.4, 39.6], color="#2f7356", lw=1.8)
arrow((84, 21.0), (84, 19.0), LINE, style="<|-|>")

# ---------------------------------------------------------------- legend / configs
ax.text(2, 4.3, "Flows:", fontsize=9.5, fontweight="bold", color=INK, va="center")
ax.plot([8.2, 11.2], [4.3, 4.3], color=FLOW_CALL, lw=2.2)
ax.text(11.8, 4.3, "tool-call path", fontsize=9, color=INK, va="center")
ax.plot([21, 24], [4.3, 4.3], color=FLOW_PRED, lw=2.2)
ax.text(24.6, 4.3, "prediction / prewarm path (C3 only)", fontsize=9, color=INK, va="center")

ax.text(2, 1.7, "Configurations:", fontsize=9.5, fontweight="bold", color=INK, va="center")
ax.text(14.2, 1.7, "C0  scale-to-zero, no prewarming        C1  always warm, 1 pod per tool        "
                   "C3  predictor + controller prewarm, TTL = 60 s idle window",
        fontsize=9, color=INK, va="center")

for ext in ("png", "svg"):
    fig.savefig(HERE / f"pulse_architecture.{ext}", dpi=200, facecolor="white")
print("saved", HERE / "pulse_architecture.png")
