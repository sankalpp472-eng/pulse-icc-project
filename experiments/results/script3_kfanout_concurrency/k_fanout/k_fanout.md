# k_fanout

## airline

| arm | trials | p50_ms | p95_ms | p99_ms | cold_calls | cold_pct | transport_failures | pod_s | pods_started | startup_s | concurrent_mean | prewarms | wasted_pct | skipped | tool_cpu_core_s | tool_mem_mib | pulse_cpu_core_s | node_cpu_busy_pct | tool_pods_peak |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| k-rule | 3 | 49.7 | 451.1 | 1452.9 | 17 | 5.8 | 0 | 10632.0 | 60 | 2.0 | 2.8 | 453 | 48.6 | 214 | 62.2 | 515.7 | 41.8 | 6.8 | 15.0 |
| k-static | 3 | 48.2 | 292.8 | 1531.2 | 23 | 7.8 | 0 | 11334.0 | 67 | 2.0 | 2.8 | 457 | 48.8 | 217 | 70.3 | 552.2 | 44.8 | 7.1 | 16.0 |

