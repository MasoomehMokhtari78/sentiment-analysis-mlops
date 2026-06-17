# Week 3 — Inference Optimization Benchmark

| Variant | Mean Latency (ms) | P50 (ms) | P95 (ms) | Throughput (req/s) | Peak Memory (MB) | Model Size (MB) | Latency Δ vs Baseline | Memory Δ vs Baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 5.878 | 5.522 | 7.568 | 33687.64 | 0.125 | 0.005 | 0.0% | 0.0% |
| batched | 6.057 | 5.784 | 7.905 | 32689.87 | 0.0 | 0.005 | -3.05% | 100.0% |
