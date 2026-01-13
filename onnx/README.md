# GLiNER2 ONNX export

- `model.onnx` (FP32 export)
- `model_int8.onnx` (dynamic INT8 quantization via onnxruntime)
- tokenizer files copied verbatim from the HF model
- a small `config.json` describing runtime constraints

The export script follows the same design:

- ONNX graph includes only encoder + span head tensor computation
- Schema logic, label mapping, and decoding stay outside the graph
- Inputs: `input_ids`, `attention_mask` (optionally `token_type_ids`)
- Output: `span_logits`
- Export with `torch.onnx.export` (opset 19) and dynamic batch/sequence axes
- Quantize with `onnxruntime.quantization.quantize_dynamic(QInt8)`

## Usage

Enter the dev shell (adds Python + ONNX deps):

```bash
nix develop
```

Install the Python dependencies with Pipenv:

```bash
cd onnx
pipenv install
cd ..
```

Export (run from the `onnx` directory so Pipenv finds the `Pipfile`):

```bash
pipenv run python export.py \
  --model-id fastino/gliner2-multi-v1 \
  --output-dir gliner2-multi-v1
```

Validation is enabled by default and compares the exported ONNX output to the
PyTorch output for a dummy batch. To skip validation or to load the quantized
model, use:

```bash
pipenv run python export.py --no-validate
pipenv run python export.py --validate-quantized
```

The output directory will include:

- `model.onnx`
- `model_int8.onnx`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `added_tokens.json`
- `spm.model`
- `config.json`

## Notes

- The export uses a fixed `max_seq_len` (default 512) and expects inputs padded
  or truncated to that length. This matches the published bundle's runtime
  config.
- The `span_logits` label axis is aligned to token positions in the input
  sequence. Use label marker token positions (`[E]`, `[C]`, `[R]`, `[L]`) to map
  logits back to schema labels. Label mapping and decoding are intentionally
  handled outside the graph.
