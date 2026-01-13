#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

from gliner2 import GLiNER2


LABEL_TOKENS = ["[E]", "[C]", "[R]", "[L]"]
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
]


class SpanLogitsWrapper(torch.nn.Module):
    def __init__(self, extractor: GLiNER2, label_token_ids):
        super().__init__()
        self.encoder = extractor.encoder
        self.span_rep = extractor.span_rep
        self.max_width = int(extractor.max_width)
        self.register_buffer(
            "label_token_ids",
            torch.tensor(label_token_ids, dtype=torch.long),
            persistent=False,
        )

    def _label_embeddings(self, token_embeddings: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        label_ids = self.label_token_ids.view(1, 1, -1)
        label_mask = (input_ids.unsqueeze(-1) == label_ids).any(dim=-1)
        return token_embeddings * label_mask.unsqueeze(-1)

    def _span_rep(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _hidden = token_embeddings.shape
        device = token_embeddings.device

        start = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, self.max_width)
        widths = torch.arange(self.max_width, device=device).unsqueeze(0).expand(seq_len, self.max_width)
        end = start + widths
        valid = end < seq_len
        start = torch.where(valid, start, torch.zeros_like(start))
        end = torch.where(valid, end, torch.zeros_like(end))

        spans_idx = torch.stack([start, end], dim=-1)
        spans_idx = spans_idx.view(1, seq_len * self.max_width, 2)
        spans_idx = spans_idx.expand(batch_size, -1, -1)
        return self.span_rep(token_embeddings, spans_idx)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        if token_type_ids is None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        token_embeddings = outputs.last_hidden_state
        span_rep = self._span_rep(token_embeddings)
        label_embs = self._label_embeddings(token_embeddings, input_ids)

        logits = torch.einsum("blkd,btd->blkt", span_rep, label_embs)
        return logits


def copy_tokenizer_files(model_id: str, output_dir: Path):
    snapshot_dir = snapshot_download(model_id, allow_patterns=TOKENIZER_FILES)
    for name in TOKENIZER_FILES:
        src = Path(snapshot_dir) / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def write_runtime_config(output_dir: Path, extractor: GLiNER2, max_seq_len: int):
    config = {
        "hidden_size": int(extractor.encoder.config.hidden_size),
        "max_width": int(extractor.max_width),
        "max_seq_len": int(max_seq_len),
        "output_format": "span_logits",
        "has_count_embed": True,
        "schema_position_included": False,
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def run_ort(session: ort.InferenceSession, inputs, include_token_type_ids: bool):
    input_ids, attention_mask = inputs[0], inputs[1]
    feed = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
    }
    if include_token_type_ids:
        feed["token_type_ids"] = inputs[2].cpu().numpy()
    outputs = session.run(["span_logits"], feed)
    return outputs[0]


def validate_onnx(onnx_path: Path, wrapper: SpanLogitsWrapper, inputs, include_token_type_ids: bool):
    session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    with torch.no_grad():
        torch_out = wrapper(*inputs).cpu().numpy()

    ort_out = run_ort(session, inputs, include_token_type_ids)
    if torch_out.shape != ort_out.shape:
        raise ValueError(f"ONNX output shape mismatch: torch {torch_out.shape} vs onnx {ort_out.shape}")
    max_abs = float(np.max(np.abs(torch_out - ort_out)))
    return max_abs


def validate_quantized(onnx_path: Path, inputs, include_token_type_ids: bool):
    session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    ort_out = run_ort(session, inputs, include_token_type_ids)
    return ort_out.shape


def export(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    label_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in LABEL_TOKENS]
    missing = [t for t, tid in zip(LABEL_TOKENS, label_token_ids) if tid is None or tid == tokenizer.unk_token_id]
    if missing:
        raise ValueError(f"Tokenizer missing special tokens: {', '.join(missing)}")

    extractor = GLiNER2.from_pretrained(args.model_id)
    extractor.eval()

    max_seq_len = args.max_seq_len
    wrapper = SpanLogitsWrapper(extractor, label_token_ids)
    wrapper.eval()

    dummy = tokenizer(
        "Apple CEO Tim Cook announced iPhone 15.",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    inputs = (input_ids, attention_mask)

    input_names = ["input_ids", "attention_mask"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "span_logits": {0: "batch", 1: "seq_len", 3: "seq_len"},
    }

    if args.include_token_type_ids:
        token_type_ids = dummy.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        inputs = (input_ids, attention_mask, token_type_ids)
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq_len"}

    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        wrapper,
        inputs,
        onnx_path.as_posix(),
        input_names=input_names,
        output_names=["span_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    if args.quantize:
        quantize_dynamic(
            onnx_path.as_posix(),
            (output_dir / "model_int8.onnx").as_posix(),
            weight_type=QuantType.QInt8,
        )

    copy_tokenizer_files(args.model_id, output_dir)
    write_runtime_config(output_dir, extractor, max_seq_len=max_seq_len)

    if args.validate:
        max_abs = validate_onnx(onnx_path, wrapper, inputs, args.include_token_type_ids)
        print(f"Validation OK (max abs diff: {max_abs:.6f})")

    if args.quantize and args.validate_quantized:
        shape = validate_quantized(output_dir / "model_int8.onnx", inputs, args.include_token_type_ids)
        print(f"Quantized model loads OK (output shape: {shape})")


def parse_args():
    parser = argparse.ArgumentParser(description="Export fastino/gliner2-multi-v1 to ONNX.")
    parser.add_argument(
        "--model-id",
        default="fastino/gliner2-multi-v1",
        help="Hugging Face model id to export.",
    )
    parser.add_argument(
        "--output-dir",
        default="onnx/gliner2-multi-v1-int8",
        help="Directory for ONNX outputs and tokenizer files.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length for export (pad inputs to this length).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=19,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--include-token-type-ids",
        action="store_true",
        help="Include token_type_ids input in the ONNX graph.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip ONNX validation after export.",
    )
    parser.add_argument(
        "--validate-quantized",
        action="store_true",
        help="Also load and run the quantized ONNX model.",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip INT8 quantization step.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.quantize = not args.no_quantize
    args.validate = not args.no_validate
    export(args)


if __name__ == "__main__":
    main()
