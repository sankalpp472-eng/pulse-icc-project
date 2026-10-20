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
      labels:            # on the template so every POD carries them (service labels are not copied to pods)
        pulse/role: mcp-tool
        pulse/tool: {tool}
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
              value: {tool}{redis_env}
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
    ap.add_argument("--image", default="docker.io/sankalps2003/pulse-mcp-tools:0.2")
    ap.add_argument("--redis-url", default="redis://redis.default.svc.cluster.local:6379/0",
                    help="shared database for all tools; '' = each pod keeps its own in-memory copy")
    ap.add_argument("--min-scale", type=int, default=0, help="0 = C0 (scale to zero), 1 = C1 (always warm)")
    ap.add_argument("--window", default="60s", help="idle time before scale-down: 60s = Knative default (min 6s)")
    ap.add_argument("-o", "--out", default=str(Path(__file__).with_name("tools.yaml")))
    a = ap.parse_args()

    tools = [t.get_info()["function"]["name"] for t in ALL_TOOLS]
    redis_env = ("\n            - name: REDIS_URL"
                 f"\n              value: {a.redis_url}") if a.redis_url else ""
    docs = [TEMPLATE.format(svc=t.replace("_", "-"), tool=t, image=a.image, min_scale=a.min_scale, window=a.window,
                            redis_env=redis_env)
            for t in tools]
    Path(a.out).write_text("---\n".join(docs), encoding="utf-8")
    print(f"wrote {a.out}: {len(tools)} services, min-scale={a.min_scale}, image={a.image}, "
          f"db={a.redis_url or 'in-memory per pod'}")


if __name__ == "__main__":
    main()
