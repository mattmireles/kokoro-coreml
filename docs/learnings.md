# Learnings from Kokoro TTS Core ML Conversion

This document captures the key challenges and solutions discovered while converting the Kokoro TTS model from PyTorch to Core ML for on-device inference on Apple Silicon.

## 1. The Core Problem: Dynamic Shapes vs. Static Graphs

The fundamental challenge was the model's heavy reliance on dynamic operations that are incompatible with Core ML's requirement for a static or predictably dynamic computation graph.

- **`torch.full` with Dynamic Inputs**: The tracer failed when creating tensors with shapes derived from dynamic inputs (e.g., `torch.full((1, input.shape[1]), ...)`).
  - **Solution**: Replace with traceable equivalents like `torch.ones_like(input).sum()`.
- **`torch.repeat_interleave`**: This was the primary blocker. The model creates an alignment matrix whose shape depends on the *values* inside the predicted duration tensor. This is impossible to represent in a static graph.
- **`pack_padded_sequence`**: The LSTMs used this for handling variable-length sequences, which is not supported by the Core ML tracer.

## 2. The Solution: A Two-Stage, Bucketed Architecture

A direct, one-to-one conversion was not feasible. The winning strategy was to re-architect the *inference pipeline* without changing the core model weights, splitting the model into two parts and using bucketing for the final stage.

### Stage 1: The `DurationModel` (Dynamic)

- **Responsibility**: Runs the expensive Transformer and LSTMs to predict phoneme durations and extract intermediate hidden states.
- **Implementation**:
  - Takes `input_ids`, `ref_s`, `speed`, and an `attention_mask` as inputs.
  - All inputs with a sequence dimension use `ct.RangeDim` to allow for variable-length text.
  - **Key Fixes**:
    - **Monkey-Patching**: We created CoreML-friendly versions of the `TextEncoder` and `DurationEncoder` in the export script. These custom modules remove the `pack_padded_sequence` calls and run the LSTMs directly on the padded tensors.
    - **BERT Buffer Removal**: We programmatically deleted the `buffered_token_type_ids` from the `AlbertModel` instance before tracing to prevent a `slice` error. The `token_type_ids` were then passed in as an input during the forward pass.
- **Output**: A set of tensors containing the predicted durations and the hidden states needed for synthesis.
- **Result**: A single, flexible `.mlpackage` that runs efficiently on the ANE.

### Stage 2: The `SynthesizerModel` (Fixed-Size Buckets)

- **Responsibility**: Takes the intermediate features and a pre-built alignment matrix and generates the final audio waveform.
- **Implementation**:
  - We created multiple `SynthesizerModel`s, each one compiled for a **fixed-size** audio output (e.g., 3s, 5s, 10s, 30s). This is known as **bucketing**.
  - By using fixed-size inputs for the alignment matrix, we completely remove the dynamic shape problem that was blocking the conversion.
- **Output**: A fixed-length audio waveform.
- **Result**: A set of highly optimized `.mlpackage` files, one for each bucket, that run entirely on the ANE.

## 3. The Client's Role: The Conductor

The complexity that was removed from the model graph is now managed by the native Swift client code. The client is responsible for:
1. Running the `DurationModel` once.
2. Summing the predicted durations to determine the final audio length.
3. Selecting the appropriate `SynthesizerModel` bucket.
4. Building the alignment matrix on the CPU (a fast, simple operation).
5. Padding the matrix to the bucket's fixed size.
6. Calling the selected `SynthesizerModel`.
7. Trimming any padding silence from the end of the final audio buffer.

## 4. Key Takeaways

- **Simpler is Better**: When faced with an impossible conversion, don't fight the tools. Redesign the *pipeline*, not the model.
- **Divide and Conquer**: Isolate dynamic, data-dependent logic from the heavy, parallelizable math.
- **CPU is Not the Enemy**: Offloading small, complex operations (like building the alignment matrix) to the CPU is a valid and powerful strategy that unlocks the ANE for the 99% of work that matters.
- **Monkey-Patching is a Powerful Tool**: For stubborn models, modifying the model instance in-memory during the export process is a clean way to fix incompatible layers without forking the original library.
- **Bucketing Beats Dynamic Hell**: When a model's output is fundamentally dynamic, creating a few fixed-size versions is often the most pragmatic path to a shippable, high-performance solution.
