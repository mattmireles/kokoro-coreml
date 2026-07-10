# iPhone Device Debug Notes

Debugging trails for running the Kokoro Core ML pipeline on physical iPhones.
Mac-side export/conversion debugging stays in [debug-notes.md](debug-notes.md);
iPhone timing evidence lives in
[iphone-performance-notes.md](iphone-performance-notes.md); device-lab
procedure is the
[iPhone Core ML device lab runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md).

One issue per section, newest first, status marked Active or Resolved.

## Issue: iPhone ANE compiler rejects the full-ANE plan (`ANECCompile() FAILED`) — Active

**Observed:** 2026-06-09, iPhone 12 Pro (A14) and iPhone 15 Pro Max (A17 Pro),
both iOS 26.5, shipped runtime `.mlpackage` set compiled to `.mlmodelc` by
Xcode 26.5.

With `computeUnits = .all` (the maximal policy; note the published Mac
Config F rows themselves run `staged` — `.all` is the historical Config F
label from
[coreml-compute-unit-ablation.md](coreml-compute-unit-ablation.md), see
[performance-notes.md](performance-notes.md) "Config F rows use the
production-shaped staged policy"), the FIRST prediction fails on both phones
with:

```
E5RT encountered an STL exception. msg = MILCompilerForANE error:
failed to compile ANE model using ANEF. Error=_ANECompiler : ANECCompile() FAILED.
```

surfacing to Swift as `com.apple.CoreML Code=0 ... (error code: -9)`.

What this rules out: ANE generation (A14 and A17 Pro fail identically while
every tested M-series Mac accepts the same packages), host-side compilation
(Xcode's coremlc succeeded; the failure is the on-device ANEF specialization),
and input data (the same frozen inputs run under other policies).

What is NOT yet isolated: which model the compiler rejects. The first predict
in the pipeline is the duration model (BERT + LSTMs), making it the prime
suspect, but the error fires before per-stage attribution is possible from
the app log alone.

**Workaround (in `ios-bench/Sources/BenchApp.swift`):** a per-bucket compute
policy ladder — `all` → `staged` → `cpuAndGPU` → `cpuOnly` — that records the
policy actually used. Both phones settle on `staged` (decoder-pre on
cpuAndNeuralEngine, everything else cpuAndGPU), which means decoder-pre's
ANEF compile SUCCEEDS on iPhone; the rejected stage is among duration, f0n,
and generator.

**Next step:** per-stage smoke test (load + one predict per model under
`.cpuAndNeuralEngine`) to name the offender, then an iPhone-targeted export
of that stage. External research on iPhone failure semantics, ANE admission
limits, and re-chunking designs is ingested in the
[Kokoro A14 iPhone generator execution guide](../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md)
(triage:
[kokoro-a14-iphone-guide-triage-2026-06-10.md](kokoro-a14-iphone-guide-triage-2026-06-10.md)).

## Issue: 4 GB iPhone jetsams the bench app (signal 9) — Resolved

**Observed:** 2026-06-09, iPhone 12 Pro (4 GB). Three kills: (1) combined run
died during MLX 7s generation with our Core ML models and MLX's 327 MB fp32
weights resident in one process; (2) 15 Pro Max (8 GB!) combined run died
right after MLX 3s under the pre-cache-cap binary; (3) MLX-only 30s run died
after one iteration even with the cache cap.

**Resolution (first two):**

- Results flush to `Documents/` after every (arm, bucket) pair, so a kill
  never loses completed buckets.
- Launch arguments `--arms`, `--keys`, `--out` split arms into separate
  processes; the MLX arm reruns standalone after the Core ML arm's process
  exits.
- `MLX.GPU.set(cacheLimit: 256 MB)` before the MLX arm stops freed GPU
  buffers from accumulating across warm iterations.

**Permanent finding (third):** MLX Swift 30s generation does not fit the
iPhone 12 Pro's foreground memory budget at fp32 even alone in the process —
one compile-inclusive iteration completes (~10.5-11.4 s), then jetsam kills
the second. Reproduced twice. Recorded as `OOM` in the published table, not
as a timing.

## Issue: app exits cleanly at launch — missing dynamic framework — Resolved

**Observed:** 2026-06-09. `devicectl` launch reported "terminated with exit
code 0" within seconds, no app output. Console showed the real story:

```
dyld: Library not loaded: @rpath/KokoroSwift.framework/KokoroSwift
```

`mlalma/kokoro-ios` declares its library product `type: .dynamic`; the
XcodeGen-generated app embedded the package's *dependencies* (MisakiSwift,
MLXUtilsLibrary, Numerics, ZIPFoundation) but not KokoroSwift.framework
itself.

**Resolution:** drop `type: .dynamic` in the vendored `Package.swift`
(`ios-bench/Vendor/kokoro-ios`) so the library links statically. Static is
the right default for a single-app consumer; dynamic only pays off shared
across extensions.

**Triage heuristic:** "exit code 0 instantly, zero output" on iOS usually
means dyld aborted before `main` — always pull the console (`devicectl
device process launch --console`) before suspecting app logic.

## Issue: launch requires unlocked device — Resolved (procedural)

Re-confirmed 2026-06-09, first documented 2026-06-06 (see
[Kokoro-M1-vocoder-boundary-research-brief.md](Kokoro-M1-vocoder-boundary-research-brief.md)):
launching via `devicectl` while the phone is locked either fails or the app
is denied a foreground scene (no Metal context, MLX arm cannot run). The
bench protocol requires: device unlocked, plugged in, app foregrounded,
idle-timer disabled (the app sets `isIdleTimerDisabled = true`).

## Issue: host `xcodebuild` stalls at CreateBuildDescription — Mitigated (host-side, blocks iPhone deploys)

This is a Mac-host blocker, not an iPhone issue, but it gates every iPhone
build so it is cross-referenced here. First documented 2026-06-06 in
[performance-notes.md](performance-notes.md) (SwiftBuild stall; reproduced
for a minimal one-file iOS app; the manual `swiftc` bypass script
`scripts/external_bakeoff/build_install_config_f_ios_manual.sh` was the
original workaround).

2026-06-09 finding: the stall is SWBBuildService's spawned `clang -v -E` SDK
probes never returning (each completes in <2 s standalone). Killing any probe
alive across two 10 s sweeps makes SwiftBuild fall back gracefully and the
build proceeds — this unblocked every `ios-bench` build, including full
mlx-swift C++ compilation. Reaper kept host-local at
`/tmp/kokoro_probe_reaper.sh` (not committed; host-specific). If it is lost,
recreate: loop `pgrep -f "clang -v -E"`, kill PIDs seen twice 10 s apart,
exit when the build log records a result.
