"""
Generate one Knative Service per MCP tool (14 tau-bench airline tools).

  python gen_services.py                      # C0: min-scale 0 (scale to zero)  -> tools.yaml
  python gen_services.py --min-scale 1 -o tools-c1.yaml   # C1: always one warm replica

Service name = tool name with "_" -> "-" (Knative names must be DNS-safe), e.g.
get_user_details -> get-user-details, reachable through Kourier with
Host: get-user-details.default.example.com and path /mcp.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
from tau_airline import ALL_TOOLS  # noqa: E402

TEMPLATE = """\
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {svc}
  namespace: default
  labels:
    pulse/role: mcp-tool
    pulse/tool: {tool}
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/min-scale: "{min_scale}"
        autoscaling.knative.dev/window: "{window}"
    spec:
      containers:
        - image: {image}
          ports:
            - containerPort: 8080
          env:
            - name: TOOLS
              value: {tool}
          # /healthz answers only once the MCP server is up (data loaded, routes mounted).
          # No periodSeconds -> Knative probes aggressively while the pod starts.
          readinessProbe:
            httpGet:
              path: /healthz
          resources:
            requests:
              cpu: 50m
              memory: 96Mi
            limits:
              memory: 256Mi
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="docker.io/sankalps2003/pulse-mcp-tools:0.1")
    ap.add_argument("--min-scale", type=int, default=0, help="0 = C0 (scale to zero), 1 = C1 (always warm)")
    ap.add_argument("--window", default="6s", help="idle window before scale-down (min 6s)")
    ap.add_argument("-o", "--out", default=str(Path(__file__).with_name("tools.yaml")))
    a = ap.parse_args()

    tools = [t.get_info()["function"]["name"] for t in ALL_TOOLS]
    docs = [TEMPLATE.format(svc=t.replace("_", "-"), tool=t, image=a.image, min_scale=a.min_scale, window=a.window)
            for t in tools]
    Path(a.out).write_text("---\n".join(docs), encoding="utf-8")
    print(f"wrote {a.out}: {len(tools)} services, min-scale={a.min_scale}, image={a.image}")


if __name__ == "__main__":
    main()
