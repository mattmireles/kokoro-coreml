# Apple Silicon NVMe And Energy Measurement Guide

This guide ingests a Deep Research Max report for macOS cache bypass and energy
measurement in SSD-tier model-weight experiments. Treat the raw report as
research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/apple-silicon-nvme-cache-bypass-and-energy-measurement-for-ssd-tier-model-benchm/2026-06-29T05-13-42-684Z/raw-report.md`

## Executive Summary

For SSD-tier MoE experiments, the benchmark must prove two things before any
predictor matters:

- reads really hit the storage device rather than the macOS file cache;
- the added reads improve tokens/sec without losing on joules/token.

macOS does not expose Linux-style `O_DIRECT` for normal file reads. The practical
path is a custom `pread` benchmark with `fcntl(..., F_NOCACHE, 1)`, conservative
cache-eviction setup, and `fs_usage -f diskio` evidence that physical disk I/O
occurred. `powermetrics` can capture estimated CPU/GPU/ANE power and thermal
state, but Apple documents those values as estimates and warns against
cross-device comparison.

The result should be treated as an envelope, not a product benchmark: if the
storage boundary cannot satisfy the oracle bandwidth ceiling, stop.

## macOS Cache And I/O Model

The local `fcntl(2)` man page documents these commands:

| API | Use | Caveat |
| --- | --- | --- |
| `F_NOCACHE` | Turns data caching off or on for a file descriptor. | It is not the same contract as Linux `O_DIRECT`; verify behavior with disk I/O traces. |
| `F_RDADVISE` | Issues an advisory async read with no copy to user. | Advisory prefetch can change the workload; use only in explicitly labeled experiments. |
| `F_RDAHEAD` | Turns read-ahead off or on. | Useful to separate explicit prefetch from kernel read-ahead. |
| `F_LOG2PHYS` | Returns disk-device information for a file offset. | Useful for diagnosing layout, not a substitute for I/O traces. |

For Stage 0, prefer normal files and explicit reads over `mmap`. Page faults
make timing harder to reason about and can obscure whether the benchmark is
measuring storage or VM behavior.

## Direct-Read Microbenchmark Pattern

Use a custom C or Swift harness instead of a generic benchmark first. The harness
should make every experimental variable explicit:

- file path and file size;
- expert block size;
- offset sequence;
- queue depth;
- read-ahead setting;
- `F_NOCACHE` result;
- wall-clock latency per read;
- bytes read per iteration;
- `fs_usage` capture path.

Minimum C shape:

```c
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static const size_t kAppleSiliconPageAlignment = 16 * 1024;

int read_expert_block(const char *path, off_t offset, size_t block_size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return -1;
    }

    int no_cache = 1;
    if (fcntl(fd, F_NOCACHE, no_cache) < 0) {
        close(fd);
        return -1;
    }

    void *buffer = NULL;
    if (posix_memalign(&buffer, kAppleSiliconPageAlignment, block_size) != 0) {
        close(fd);
        return -1;
    }

    ssize_t n = pread(fd, buffer, block_size, offset);
    free(buffer);
    close(fd);
    return n == (ssize_t)block_size ? 0 : -1;
}
```

Generate test files from incompressible bytes. Do not use zero-filled sparse
files; APFS compression or sparse allocation can turn an SSD test into a
filesystem metadata test.

Run separate cells for:

- sequential offsets;
- random offsets;
- queue depth 1;
- the queue depths needed by the proposed prefetch worker;
- cold run after cache eviction;
- warmed run where cache hits are expected and labeled.

## Proving Physical Device Reads

User-space timings are not proof. Capture `fs_usage` beside each run:

```bash
sudo fs_usage -w -f diskio <pid> > outputs/moe_prefetch/fs_usage_diskio.txt
```

The `fs_usage(1)` man page says `-f diskio` filters disk I/O events and that
wide mode includes columns such as byte count, disk block number, offset, time
interval, and process name. Preserve the raw trace for any claimed result.

Use `iostat` only as a coarse whole-system sanity check:

```bash
iostat -w 1
```

It is not enough by itself because it is not process-scoped and cannot explain
which read was served by cache.

## Energy And Thermal Procedure

Use `powermetrics` for sustained captures, not one-shot timing around a single
token. Local help on this machine lists these relevant samplers:

- `disk`
- `cpu_power`
- `gpu_power`
- `ane_power`
- `thermal`

Example capture:

```bash
sudo powermetrics \
  --samplers disk,cpu_power,gpu_power,ane_power,thermal \
  --sample-rate 500 \
  --format plist \
  --output-file outputs/moe_prefetch/powermetrics.plist
```

Record the exact command, macOS version, machine model, AC/battery state, and
sample interval. `powermetrics` reports estimated power values; use them for
within-machine comparisons and placement signals, not cross-device rankings.

For joules/token:

1. Align benchmark start and stop timestamps with the power capture.
2. Integrate average power over the decode window.
3. Divide by decoded tokens.
4. Report median and IQR across repeated runs.
5. Discard or separately label runs with non-nominal thermal state.

## Common False Positives

| False Positive | Symptom | Fix |
| --- | --- | --- |
| File cache hit | Read bandwidth looks like DRAM, not SSD. | Use `F_NOCACHE`, run cold/warm cells separately, and preserve `fs_usage -f diskio` traces. |
| Sparse or compressed dummy file | Impossible bandwidth from tiny physical reads. | Fill test files with random bytes and record physical file allocation if needed. |
| Kernel read-ahead hiding random-read cost | Sequential run looks good, random production run stalls. | Turn read-ahead off for one cell and benchmark random expert offsets. |
| UMA contention | SSD prefetch benchmark wins alone but slows inference. | Measure concurrent compute plus prefetch, not storage in isolation only. |
| Thermal drift | Later iterations slow down or power state changes. | Use quiet-host gates, cooldowns, and `powermetrics --samplers thermal`. |
| One lucky run | A single row supports the claim. | Use repeated runs with median and IQR. |

## Stage 0 Envelope

Stage 0 should output:

- expert byte size at the chosen quantization;
- measured cold sequential bandwidth at expert block size;
- measured cold random bandwidth at expert block size;
- p50/p95 read latency per expert block;
- one-layer compute time for the target MoE harness;
- hideability ratio: `expert_fetch_latency / one_layer_compute_time`;
- oracle bandwidth ceiling: `active_expert_bytes_per_token / measured_nvme_bandwidth`;
- `fs_usage` evidence path;
- go/kill decision.

If the oracle ceiling is below the acceptable target tokens/sec, stop. The
correct next move is compression, smaller experts, fewer active experts, or a
different model. A learned prefetcher cannot beat physics.

## Stage 3 Evidence

A real implementation result needs all of this beside the headline number:

- demand-paging baseline with the same cache cap;
- learned policy result;
- best trivial policy result;
- oracle simulation result;
- `fs_usage` trace for each policy;
- `powermetrics` capture for each policy;
- SSD bytes read/token;
- wasted prefetched bytes/token;
- tokens/sec median and IQR;
- joules/token median and IQR;
- thermal-state log;
- simulator-vs-real delta.

## Related Documentation

- [MoE expert offload and prefetch prior art](../moe-expert-offload-prefetch-prior-art-guide.md)
- [Apple Silicon warmed-inference benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Core ML compute-unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [MoE SSD/DRAM guide triage note](../../Notes/moe-ssd-dram-prefetch-guide-triage-2026-06-29.md)
- [Core ML compute-unit ablation](../../Notes/coreml-compute-unit-ablation.md)

## Works Cited

1. [Apple `fcntl(2)` manual page](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fcntl.2.html) - documents `F_NOCACHE`, `F_RDADVISE`, `F_RDAHEAD`, and related commands.
2. Local `fs_usage(1)` man page - documents `-f diskio`, `-f cachehit`, byte-count, disk-block, and offset columns.
3. Local `powermetrics(1)` man page and `powermetrics -h` - documents samplers, plist format, and the warning that power values are estimates.
4. [Core ML compute-unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md) - existing repo guidance for interpreting `powermetrics` as placement evidence.
