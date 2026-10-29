"""
Generate one Knative Service per MCP tool (14 tau-bench airline tools, or the 16 retail ones).

  python gen_services.py                      # C0: min-scale 0 (scale to zero)  -> tools.yaml
  python gen_services.py --domain retail      # retail tools -> tools-retail.yaml
  python gen_services.py --min-scale 1 -o tools-c1.yaml   # C1: always one warm replica
  python gen_services.py --concurrency 0      # old setup: one pod takes any number of requests at once

Scaling: every pod serves --concurrency requests at a time (default 1, Knative containerConcurrency).
Extra requests wait in the activator / queue-proxy, and the autoscaler (KPA) asks for
ceil(in-flight requests / concurrency) pods, so a burst of N parallel calls to one tool starts N pods
(up to --max-scale). With --concurrency 0 (no limit) one pod takes the whole burst.

Service name = tool name with "_" -> "-" (Knative names must be DNS-safe), e.g.
get_user_details -> get-user-details, reachable through Kourier with
Host: get-user-details.default.example.com and path /mcp.

Retail (--domain retail): service names start with "retail-" (both domains have get_user_details,
calculate, think and transfer_to_human_agents), e.g. retail-get-order-details; the pods carry
pulse/role=mcp-tool-retail, so the airline runners (which watch pulse/role=mcp-tool) never count
them; the tools get DOMAIN=retail and their own Redis database (.../1, both domains have "users").
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
import tau_airline  # noqa: E402
import tau_retail  # noqa: E402

PACKAGES = {"airline": tau_airline, "retail": tau_retail}

TEMPLATE = """\
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: {svc}
  namespace: default
  labels:
    pulse/role: {role}
    pulse/tool: {tool}
spec:
  template:
    metadata:
      labels:            # on the template so every POD carries them (service labels are not copied to pods)
        pulse/role: {role}
        pulse/tool: {tool}
      annotations:
        autoscaling.knative.dev/min-scale: "{min_scale}"
        autoscaling.knative.dev/window: "{window}"{scaling}
    spec:
      containerConcurrency: {concurrency}   # requests one pod serves at once (0 = no limit)
      containers:
        - image: {image}
          ports:
            - containerPort: 8080
          env:
            - name: TOOLS
              value: {tool}{domain_env}{redis_env}
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
    ap.add_argument("--domain", choices=sorted(PACKAGES), default="airline")
    ap.add_argument("--image", default=None,
                    help="default docker.io/sankalps2003/pulse-mcp-tools:0.2 (airline) / :0.3 (retail; "
                         "0.2 has no retail tools)")
    ap.add_argument("--redis-url", default=None,
                    help="shared database for all tools (default redis://redis.default.svc.cluster.local:6379/0, "
                         "/1 for retail); '' = each pod keeps its own in-memory copy")
    ap.add_argument("--min-scale", type=int, default=0, help="0 = C0 (scale to zero), 1 = C1 (always warm)")
    ap.add_argument("--window", default="60s", help="idle time before scale-down: 60s = Knative default (min 6s)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests one pod serves at once (containerConcurrency); 1 = a burst of N calls needs N pods, "
                         "0 = no limit (one pod takes all)")
    ap.add_argument("--max-scale", type=int, default=10,
                    help="most pods per tool (0 = no limit); keeps a burst within the cluster's capacity")
    ap.add_argument("-o", "--out", default=None, help="default tools.yaml (airline) / tools-retail.yaml")
    a = ap.parse_args()

    retail = a.domain == "retail"
    if a.redis_url is None:
        a.redis_url = f"redis://redis.default.svc.cluster.local:6379/{1 if retail else 0}"
    a.image = a.image or f"docker.io/sankalps2003/pulse-mcp-tools:{'0.3' if retail else '0.2'}"
    a.out = a.out or str(Path(__file__).with_name("tools-retail.yaml" if retail else "tools.yaml"))
    prefix, role = ("retail-", "mcp-tool-retail") if retail else ("", "mcp-tool")
    domain_env = ("\n            - name: DOMAIN"
                  "\n              value: retail") if retail else ""

    tools = [t.get_info()["function"]["name"] for t in PACKAGES[a.domain].ALL_TOOLS]
    redis_env = ("\n            - name: REDIS_URL"
                 f"\n              value: {a.redis_url}") if a.redis_url else ""
    # target-utilization 100: scale at 1 in-flight request per pod, not at 70% of it (Knative default)
    scaling = "".join(f'\n        autoscaling.knative.dev/{k}: "{v}"' for k, v in
                      ([("max-scale", a.max_scale)] if a.max_scale else []) +
                      ([("target-utilization-percentage", 100)] if a.concurrency else []))
    docs = [TEMPLATE.format(svc=prefix + t.replace("_", "-"), tool=t, role=role, image=a.image,
                            min_scale=a.min_scale, window=a.window, domain_env=domain_env,
                            redis_env=redis_env, scaling=scaling, concurrency=a.concurrency)
            for t in tools]
    Path(a.out).write_text("---\n".join(docs), encoding="utf-8")
    print(f"wrote {a.out}: {len(tools)} {a.domain} services, min-scale={a.min_scale}, "
          f"concurrency={a.concurrency or 'unlimited'}, max-scale={a.max_scale or 'unlimited'}, image={a.image}, "
          f"db={a.redis_url or 'in-memory per pod'}")


if __name__ == "__main__":
    main()
