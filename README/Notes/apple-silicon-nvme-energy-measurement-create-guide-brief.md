# Create-Guide Brief: Apple Silicon NVMe And Energy Measurement

## Topic

Defeating macOS caching and measuring real NVMe I/O latency, bandwidth, power, thermals, and joules/token on Apple Silicon for SSD-tier model-weight experiments.

## Target Repo

`kokoro-coreml`

## Target Guide Path

`README/Guides/apple-silicon/apple-silicon-nvme-energy-measurement-guide.md`

## Context

We are evaluating a proposed learned SSD-to-DRAM expert prefetcher for on-device MoE inference. Stage 0 is intentionally simple: prove whether the physical SSD/DRAM boundary is hideable before implementing any predictor. The biggest risk is measuring the macOS page cache instead of NVMe, or measuring performance without energy and thermal controls.

The guide should be a practical field manual for an engineer writing direct I/O microbenchmarks and benchmark harnesses on Apple Silicon. It should explain the actual macOS APIs and tools, their limitations, and how to prove that reads hit the device.

## Primary Research Goal

Establish the most reliable way to measure expert-sized SSD reads, bypass or evict file cache on macOS, verify device I/O, and compute energy per token for model-weight paging experiments on Apple Silicon.

## Questions To Answer

- What does `fcntl(fd, F_NOCACHE, 1)` actually guarantee on macOS? What does it not guarantee?
- How should a benchmark use `pread`, aligned buffers, file layout, queue depth, random versus sequential access, and repeated runs to approximate expert-sized reads?
- Are `O_DIRECT`, `posix_fadvise`, `F_RDADVISE`, `F_NOCACHE`, `mmap`, `msync`, `purge`, and `fcntl` relevant on macOS? Which are available and reliable?
- How can an engineer verify that reads hit NVMe rather than page cache? Cover `fs_usage`, `iostat`, Activity Monitor, DTrace alternatives, Instruments, and any Apple-supported approaches.
- What are the limitations of `fio` on macOS for direct I/O and cache bypass? Is a custom C/Swift direct-read loop preferable?
- How should Stage 0 measure expert-granularity latency and bandwidth on Apple Silicon?
- How should Stage 3 measure energy and joules/token with `powermetrics`? Include required privileges, sampling intervals, fields to collect, and caveats.
- How should thermal state, battery/AC power, Spotlight, Time Machine, file compression, APFS behavior, and background I/O be controlled?
- What differences matter between Apple Silicon UMA and discrete Linux/NVMe/GPU systems?
- What minimum evidence should be recorded before claiming an SSD prefetch win?

## Source Hints

Prioritize Apple documentation, `man` pages, Darwin/XNU references, `fio` docs, WWDC/Instruments documentation, and reliable systems posts. Use third-party posts only when primary docs are insufficient, and label them as such.

Specific search targets:

- `fcntl F_NOCACHE macOS`
- `man fcntl F_NOCACHE Darwin`
- `fs_usage macOS file system reads`
- `iostat macOS disk bandwidth`
- `powermetrics Apple Silicon ANE GPU CPU energy`
- `fio macOS direct io`
- `APFS cache benchmark macOS`

## Output Format

Produce a text-only Markdown field guide. Do not include charts, generated images, diagrams, or visualizations.

Required sections:

1. Executive summary for Stage 0 and Stage 3 experiment design.
2. macOS cache model and what each API does.
3. Recommended direct-read microbenchmark design, with C or Swift pseudocode.
4. How to verify physical device reads.
5. Energy and thermal measurement procedure.
6. Common false positives and how to catch them.
7. Linux portability comparison, limited to what differs from Apple Silicon.
8. Minimum evidence checklist for accepting a result.
9. Source list with direct links.

Mark speculative recommendations separately from evidence-backed findings.
