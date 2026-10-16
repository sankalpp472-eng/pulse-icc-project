"""
Kill-pod cold-start probe (the closest OpenFaaS CE gets to the faasd method).

CE never lets a function sit at 0 replicas, so instead we delete the pod and
call the function immediately. Kubernetes restarts the pod right away; the
gateway does NOT hold the request, so many calls fail (500/503) instead of
waiting -- that failure is itself the finding.

  python kill_pod_probe.py 5      # 5 trials
"""
import json, subprocess, sys, time, urllib.request, statistics
NS, FN, GW = "openfaas-fn", "cold-tool", "http://127.0.0.1:8080"
k = lambda *a: subprocess.run(["kubectl", *a], capture_output=True, text=True).stdout
def call():
    t = time.perf_counter()
    try:
        b = json.loads(urllib.request.urlopen(urllib.request.Request(f"{GW}/function/{FN}", b"{}", method="POST"), timeout=60).read())
    except Exception as e:
        b = {"err": str(e)[:80]}
    return (time.perf_counter() - t) * 1000, b
cold, warm = [], []
for i in range(int(sys.argv[1])):
    k("delete", "pod", "-n", NS, "-l", f"faas_function={FN}", "--wait=false")
    ms, b = call()
    cold.append(ms); print(f"  cold {ms:7.0f} ms  first_request={b.get('first_request')} {b.get('err','')}")
    warm += [call()[0] for _ in range(3)]
    time.sleep(3)
print(f"  cold median {statistics.median(cold):.0f} ms (min {min(cold):.0f}, max {max(cold):.0f}) | warm median {statistics.median(warm):.0f} ms")
