# Core ML Compute Plans for Enumerated-Shape Specializations

This field guide answers one narrow question: how do you obtain honest
`MLComputePlan` evidence for a non-default `EnumeratedShapes` bucket when the
public compute-plan API has no shape selector?

It ingests and corrects this external Max research report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-mlcomputeplan-for-non-default-enumeratedshapes-specializations/2026-07-21T04-19-00-642Z/raw-report.md`

The raw report is research input, not canonical truth. Its claim that
`MLComputePlan` and `materialize_dynamic_shape_mlmodel` require coremltools 9.0
is false for this repo's frozen environment. Both APIs exist in coremltools
8.3.0. Its example also omitted the required destination path and treated a
mutating utility as though it returned a model.

## The constraint

An enumerated input defines a finite set of legal shapes and one default
shape. If no default is supplied, the first enumerated shape is the default.
At runtime, prediction selects a legal specialization from actual input
shapes.

`MLComputePlan` is different. Apple's public Swift load method takes only the
compiled-model URL and `MLModelConfiguration`. The Python wrapper likewise has
no input tensor, shape map, bucket, or specialization-index argument. A plan
loaded from a flexible package therefore cannot be presented as proof for an
arbitrary non-default bucket.

This distinction matters whenever placement changes with tensor size. A plan
for the 128-token default does not prove where the 512-token specialization is
preferred to run.

## The smallest honest solution

When a gate requires bucket-specific per-operation placement, materialize a
fixed-shape diagnostic twin for that bucket and capture the compute plan from
the twin.

```python
from pathlib import Path

import coremltools as ct

source = ct.models.MLModel(
    "segment_00_conv_0_1.mlpackage",
    skip_model_load=True,
)
destination = Path("segment_00_conv_0_1_fixed512.mlpackage")

ct.models.utils.materialize_dynamic_shape_mlmodel(
    dynamic_shape_mlmodel=source,
    function_name_to_materialization_map={
        "main": {
            "hidden_states": (1, 1024, 1, 512),
            "conv_state_0_in": (1, 1024, 2),
            "conv_state_1_in": (1, 1024, 2),
        }
    },
    destination_path=str(destination),
)

fixed = ct.models.MLModel(str(destination), skip_model_load=True)
```

In coremltools 8.3.0 the verified signature is:

```text
materialize_dynamic_shape_mlmodel(
    dynamic_shape_mlmodel,
    function_name_to_materialization_map,
    destination_path,
    source_function_name="main",
) -> None
```

Supply every flexible input in the shape map. For synchronized multi-input
enumeration, the shapes must all describe the same registered bucket.

## A materialized twin is a new artifact

The utility loads the source MIL program, applies a symbolic-shape
materialization pass, reruns the default optimization pipeline, and writes a
new package. It can therefore change serialization, graph structure, fusion,
and package-tree SHA-256 even when predictions remain numerically identical.

Use this evidence chain:

1. Hash the original enumerated package.
2. Materialize one explicitly named fixed bucket.
3. Verify that no flexible input remains and every input shape is exact.
4. Hash the fixed twin independently.
5. Compare source and twin outputs on the same real bucket input.
6. Capture `MLComputePlan` from the fixed twin under every claim-bearing
   `computeUnits` policy.
7. Bind timing to that exact fixed-twin hash if the plan decides a timing gate.

Do not claim that the fixed twin has the source package's identity. Do not
profile the twin and time the flexible source when a strict one-artifact gate
is required.

## Python compute-plan capture

After compiling the fixed package, request the exact compute-unit policy whose
placement enters the report:

```python
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan

configuration = ct.models.MLModelConfiguration()
configuration.compute_units = ct.ComputeUnit.CPU_AND_NE

plan = MLComputePlan.load_from_path(
    "segment_00_conv_0_1_fixed512.mlmodelc",
    configuration,
)
```

Record at minimum:

- fixed package-tree SHA-256;
- requested compute-unit policy;
- host/device identity and OS build;
- coremltools version;
- total costed operations;
- preferred-device histogram;
- per-operator-type device histogram.

Placement is admission evidence, not latency or correctness evidence. Run
real inputs under the same policy and keep numerical gates ahead of timing.

## When not to materialize

Do not create diagnostic twins merely because a model has flexible shapes.
They are justified when all of the following hold:

- the claimed bucket is not the package default;
- bucket-specific placement can change the decision;
- the public plan API offers no specialization selector;
- the experiment can bind validation, plan evidence, and timing to the twin.

If the question is only which default specialization the compiler prefers,
the original package is enough. If Instruments can directly corroborate the
runtime bucket, retain that trace as runtime evidence, but do not invent a
programmatic plan selector that Apple does not expose.

## Primary references

- [Apple coremltools model utilities API](https://apple.github.io/coremltools/source/coremltools.models.html)
- [Apple flexible input shapes guide](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)
- [Apple MLComputePlan load API](https://developer.apple.com/documentation/coreml/mlcomputeplan-1w21n/load%28contentsof%3Aconfiguration%3A%29)
- [coremltools utility implementation](https://apple.github.io/coremltools/_modules/coremltools/models/utils.html)

## Related documentation

- [LFM2.5 surgical prefill export](LFM2-surgical-prefill-CoreML-guide.md)
- [Core ML compute-unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [Warmed-inference benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Selective prefill terminal result](../../Notes/lfm2-selective-split-result.md)
