#!/usr/bin/env python3
import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from huggingface_hub import snapshot_download
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from gliner2 import GLiNER2

from decoder import SpanLogitsDecoder
from model import SpanLogitsWrapper
from validation import validate_extraction_methods, validate_onnx, validate_quantized


LABEL_TOKENS = ["[E]", "[C]", "[R]", "[L]"]
DUMMY_TEXT = "Apple CEO Tim Cook announced iPhone 15."
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
]


Tensor = torch.Tensor
InputTensors = Tuple[Tensor, ...]
DynamicAxes = Dict[str, Dict[int, str]]


@dataclass(frozen=True)
class ExportConfig:
    model_id: str
    output_dir: Path
    max_seq_len: int
    opset: int
    include_token_type_ids: bool
    quantize: bool
    validate: bool
    validate_quantized: bool
    validate_extraction: bool


def copy_tokenizer_files(model_id: str, output_dir: Path) -> None:
    snapshot_dir = snapshot_download(model_id, allow_patterns=TOKENIZER_FILES)
    for name in TOKENIZER_FILES:
        src = Path(snapshot_dir) / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def write_runtime_config(
    output_dir: Path, extractor: GLiNER2, max_seq_len: int
) -> None:
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


def label_token_ids_for(
    tokenizer: PreTrainedTokenizerBase, tokens: Sequence[str]
) -> List[int]:
    ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    missing = [
        token
        for token, token_id in zip(tokens, ids)
        if token_id is None or token_id == tokenizer.unk_token_id
    ]
    if missing:
        raise ValueError(f"Tokenizer missing special tokens: {', '.join(missing)}")
    return ids


def build_inputs(
    tokenizer: PreTrainedTokenizerBase,
    max_seq_len: int,
    include_token_type_ids: bool,
) -> Tuple[InputTensors, List[str], DynamicAxes]:
    dummy = tokenizer(
        DUMMY_TEXT,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_seq_len,
    )
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]
    inputs: InputTensors = (input_ids, attention_mask)

    input_names = ["input_ids", "attention_mask"]
    dynamic_axes: DynamicAxes = {
        "input_ids": {0: "batch", 1: "seq_len"},
        "attention_mask": {0: "batch", 1: "seq_len"},
        "span_logits": {0: "batch", 1: "seq_len", 3: "seq_len"},
    }

    if include_token_type_ids:
        token_type_ids = dummy.get("token_type_ids")
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        inputs = (input_ids, attention_mask, token_type_ids)
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq_len"}

    return inputs, input_names, dynamic_axes


def export_onnx(
    onnx_path: Path,
    model: torch.nn.Module,
    inputs: InputTensors,
    input_names: Sequence[str],
    dynamic_axes: DynamicAxes,
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        inputs,
        onnx_path.as_posix(),
        input_names=list(input_names),
        output_names=["span_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def export(config: ExportConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)
    label_token_ids = label_token_ids_for(tokenizer, LABEL_TOKENS)

    extractor = GLiNER2.from_pretrained(config.model_id)
    extractor.eval()

    wrapper = SpanLogitsWrapper(extractor, label_token_ids)
    wrapper.eval()

    inputs, input_names, dynamic_axes = build_inputs(
        tokenizer,
        max_seq_len=config.max_seq_len,
        include_token_type_ids=config.include_token_type_ids,
    )

    onnx_path = config.output_dir / "model.onnx"
    export_onnx(
        onnx_path,
        wrapper,
        inputs,
        input_names=input_names,
        dynamic_axes=dynamic_axes,
        opset=config.opset,
    )

    if config.quantize:
        quantize_dynamic(
            onnx_path.as_posix(),
            (config.output_dir / "model_int8.onnx").as_posix(),
            weight_type=QuantType.QInt8,
        )

    copy_tokenizer_files(config.model_id, config.output_dir)
    write_runtime_config(config.output_dir, extractor, max_seq_len=config.max_seq_len)

    if config.validate:
        max_abs = validate_onnx(
            onnx_path.as_posix(), wrapper, inputs, config.include_token_type_ids
        )
        print(f"Validation OK (max abs diff: {max_abs:.6f})")

    if config.quantize and config.validate_quantized:
        shape = validate_quantized(
            (config.output_dir / "model_int8.onnx").as_posix(),
            inputs,
            config.include_token_type_ids,
        )
        print(f"Quantized model loads OK (output shape: {shape})")

    if config.validate and config.validate_extraction:
        decoder = SpanLogitsDecoder(tokenizer, extractor.max_width, config.max_seq_len)
        validate_extraction_methods(
            onnx_path.as_posix(),
            wrapper,
            decoder,
            config.include_token_type_ids,
        )
        print("Extraction validation OK (entities, classification, json)")


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(
        description="Export fastino/gliner2-multi-v1 to ONNX."
    )
    parser.add_argument(
        "--model-id",
        default="fastino/gliner2-multi-v1",
        help="Hugging Face model id to export.",
    )
    parser.add_argument(
        "--output-dir",
        default="onnx/gliner2-multi-v1",
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
    parser.add_argument(
        "--no-validate-extraction",
        action="store_true",
        help="Skip extraction-method validation.",
    )
    args = parser.parse_args()
    return ExportConfig(
        model_id=args.model_id,
        output_dir=Path(args.output_dir),
        max_seq_len=args.max_seq_len,
        opset=args.opset,
        include_token_type_ids=args.include_token_type_ids,
        quantize=not args.no_quantize,
        validate=not args.no_validate,
        validate_quantized=args.validate_quantized,
        validate_extraction=not args.no_validate_extraction,
    )


def main() -> None:
    export(parse_args())


if __name__ == "__main__":
    main()
