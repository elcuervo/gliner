#!/usr/bin/env python3
import argparse
import json
import shutil

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch

from huggingface_hub import snapshot_download
from onnxruntime.quantization import quantize_dynamic, QuantType
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from gliner2 import GLiNER2


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


@dataclass(frozen=True)
class PreparedInputs:
    input_ids: Tensor
    attention_mask: Tensor
    label_positions: List[int]
    pos_to_word_index: List[Optional[int]]
    start_map: List[int]
    end_map: List[int]
    text_len: int
    original_text: str


@dataclass(frozen=True)
class EntitiesCase:
    text: str
    labels: Sequence[str]
    descriptions: Mapping[str, str]
    threshold: float = 0.5


@dataclass(frozen=True)
class ClassificationCase:
    text: str
    task_name: str
    labels: Sequence[str]
    threshold: float = 0.5
    multi_label: bool = False


@dataclass(frozen=True)
class JsonCase:
    text: str
    parent: str
    fields: Sequence[str]
    threshold: float = 0.5


class SpanLogitsWrapper(torch.nn.Module):
    def __init__(self, extractor: GLiNER2, label_token_ids: Sequence[int]):
        super().__init__()
        self.encoder = extractor.encoder
        self.span_rep = extractor.span_rep
        self.max_width = int(extractor.max_width)
        self.register_buffer(
            "label_token_ids",
            torch.tensor(label_token_ids, dtype=torch.long),
            persistent=False,
        )

    def _label_embeddings(self, token_embeddings: Tensor, input_ids: Tensor) -> Tensor:
        label_ids = self.label_token_ids.view(1, 1, -1)
        label_mask = (input_ids.unsqueeze(-1) == label_ids).any(dim=-1)
        return token_embeddings * label_mask.unsqueeze(-1)

    def _span_rep(self, token_embeddings: Tensor) -> Tensor:
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

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Optional[Tensor] = None,
    ) -> Tensor:
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


def copy_tokenizer_files(model_id: str, output_dir: Path) -> None:
    snapshot_dir = snapshot_download(model_id, allow_patterns=TOKENIZER_FILES)
    for name in TOKENIZER_FILES:
        src = Path(snapshot_dir) / name
        if src.exists():
            shutil.copy2(src, output_dir / name)


def write_runtime_config(output_dir: Path, extractor: GLiNER2, max_seq_len: int) -> None:
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


def run_ort(session: ort.InferenceSession, inputs: InputTensors, include_token_type_ids: bool) -> np.ndarray:
    input_ids, attention_mask = inputs[0], inputs[1]
    feed = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
    }
    if include_token_type_ids:
        feed["token_type_ids"] = inputs[2].cpu().numpy()
    outputs = session.run(["span_logits"], feed)
    return outputs[0]


def validate_onnx(
    onnx_path: Path,
    wrapper: SpanLogitsWrapper,
    inputs: InputTensors,
    include_token_type_ids: bool,
) -> float:
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


def validate_quantized(
    onnx_path: Path,
    inputs: InputTensors,
    include_token_type_ids: bool,
) -> Tuple[int, ...]:
    session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    ort_out = run_ort(session, inputs, include_token_type_ids)
    return ort_out.shape


class SpanLogitsDecoder:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_width: int, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_width = int(max_width)
        self.max_seq_len = int(max_seq_len)
        self.pre_tokenizer = BertPreTokenizer()

    def normalize_text(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return "."
        if normalized.endswith((".", "!", "?")):
            return normalized
        return f"{normalized}."

    def split_words(self, text: str) -> Tuple[List[str], List[int], List[int]]:
        tokens: List[str] = []
        starts: List[int] = []
        ends: List[int] = []
        for token, (start, end) in self.pre_tokenizer.pre_tokenize_str(text):
            normalized = token.lower()
            if not normalized:
                continue
            tokens.append(normalized)
            starts.append(start)
            ends.append(end)
        return tokens, starts, ends

    def build_prompt(self, base: str, label_descriptions: Mapping[str, str]) -> str:
        prompt = base
        for label, desc in label_descriptions.items():
            if desc:
                prompt += f" [DESCRIPTION] {label}: {desc}"
        return prompt

    def schema_tokens_for(self, prompt: str, labels: Sequence[str], label_prefix: str) -> List[str]:
        tokens = ["(", "[P]", prompt, "("]
        for label in labels:
            tokens.extend([label_prefix, str(label)])
        tokens.extend([")", ")"])
        return tokens

    def encode_pretokenized(self, tokens: Sequence[str]) -> Tuple[List[int], List[Optional[int]]]:
        enc = self.tokenizer(
            list(tokens),
            is_split_into_words=True,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return enc["input_ids"], enc.word_ids()

    def truncate_inputs(self, input_ids: List[int], word_ids: List[Optional[int]]) -> Tuple[List[int], List[Optional[int]]]:
        if len(input_ids) <= self.max_seq_len:
            return input_ids, word_ids
        return input_ids[: self.max_seq_len], word_ids[: self.max_seq_len]

    def build_pos_to_word_index(self, word_ids: List[Optional[int]], text_start_combined: int) -> List[Optional[int]]:
        mapping: List[Optional[int]] = [None] * len(word_ids)
        seen: Dict[int, bool] = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in seen:
                continue
            seen[word_id] = True
            if word_id >= text_start_combined:
                mapping[idx] = word_id - text_start_combined
        return mapping

    def infer_effective_text_len(self, word_ids: List[Optional[int]], text_start_combined: int, full_text_len: int) -> int:
        present = [wid for wid in word_ids if wid is not None and wid >= text_start_combined]
        if not present:
            return full_text_len
        max_text_wid = max(present)
        return min((max_text_wid - text_start_combined) + 1, full_text_len)

    def label_positions_for(self, word_ids: List[Optional[int]], label_count: int) -> List[int]:
        positions: List[int] = []
        for idx in range(label_count):
            combined_idx = 4 + (idx * 2)
            try:
                pos = word_ids.index(combined_idx)
            except ValueError as exc:
                raise ValueError(f"Could not locate label position at combined index {combined_idx}") from exc
            positions.append(pos)
        return positions

    def prepare_inputs(
        self,
        text: str,
        schema_tokens: Sequence[str],
        label_count: int,
        already_normalized: bool = False,
    ) -> PreparedInputs:
        normalized_text = text if already_normalized else self.normalize_text(text)
        words, start_map, end_map = self.split_words(normalized_text)
        combined_tokens = list(schema_tokens) + ["[SEP_TEXT]"] + words

        input_ids, word_ids = self.encode_pretokenized(combined_tokens)
        input_ids, word_ids = self.truncate_inputs(input_ids, word_ids)

        text_start_combined = len(schema_tokens) + 1
        text_len = self.infer_effective_text_len(word_ids, text_start_combined, len(words))
        pos_to_word_index = self.build_pos_to_word_index(word_ids, text_start_combined)
        label_positions = self.label_positions_for(word_ids, label_count)

        return PreparedInputs(
            input_ids=torch.tensor([input_ids], dtype=torch.long),
            attention_mask=torch.tensor([[1] * len(input_ids)], dtype=torch.long),
            label_positions=label_positions,
            pos_to_word_index=pos_to_word_index,
            start_map=start_map,
            end_map=end_map,
            text_len=text_len,
            original_text=normalized_text,
        )

    def sigmoid(self, value: float) -> float:
        return 1.0 / (1.0 + np.exp(-value))

    def find_spans_for_label(
        self,
        logits: np.ndarray,
        label_pos: int,
        prepared: PreparedInputs,
        threshold: float,
    ) -> List[Tuple[str, float, int, int]]:
        spans: List[Tuple[str, float, int, int]] = []
        seq_len = logits.shape[1]
        for pos in range(seq_len):
            start_word = prepared.pos_to_word_index[pos]
            if start_word is None:
                continue
            for width in range(self.max_width):
                end_word = start_word + width
                if end_word >= prepared.text_len:
                    continue
                score = self.sigmoid(float(logits[0, pos, width, label_pos]))
                if score < threshold:
                    continue
                char_start = prepared.start_map[start_word]
                char_end = prepared.end_map[end_word]
                text_span = prepared.original_text[char_start:char_end].strip()
                if not text_span:
                    continue
                spans.append((text_span, score, char_start, char_end))
        return spans

    def choose_best_span(self, spans: List[Tuple[str, float, int, int]]) -> Optional[Tuple[str, float, int, int]]:
        if not spans:
            return None
        sorted_spans = sorted(spans, key=lambda item: (-item[1], item[3] - item[2], len(item[0])))
        best = sorted_spans[0]
        best_score = best[1]
        near = [span for span in sorted_spans if (best_score - span[1]) <= 0.02]
        if not near:
            return best
        return min(near, key=lambda item: (item[3] - item[2], -item[1], len(item[0])))

    def format_spans(self, spans: List[Tuple[str, float, int, int]]) -> List[str]:
        if not spans:
            return []
        sorted_spans = sorted(spans, key=lambda item: -item[1])
        selected: List[Tuple[str, float, int, int]] = []
        for text, score, start_pos, end_pos in sorted_spans:
            overlaps = any(not (end_pos <= s or start_pos >= e) for _, _, s, e in selected)
            if overlaps:
                continue
            selected.append((text, score, start_pos, end_pos))
        return [text for text, _score, _start, _end in selected]

    def extract_entities(
        self,
        logits: np.ndarray,
        prepared: PreparedInputs,
        labels: Sequence[str],
        threshold: float,
    ) -> Dict[str, List[str]]:
        entities: Dict[str, List[str]] = {}
        for label_index, label in enumerate(labels):
            label_pos = prepared.label_positions[label_index]
            spans = self.find_spans_for_label(logits, label_pos, prepared, threshold)
            entities[str(label)] = self.format_spans(spans)
        return {"entities": entities}

    def classification_scores(
        self,
        logits: np.ndarray,
        prepared: PreparedInputs,
        labels: Sequence[str],
    ) -> List[float]:
        scores: List[float] = []
        for label_index in range(len(labels)):
            label_pos = prepared.label_positions[label_index]
            best = -float("inf")
            seq_len = logits.shape[1]
            for pos in range(seq_len):
                start_word = prepared.pos_to_word_index[pos]
                if start_word is None:
                    continue
                for width in range(self.max_width):
                    end_word = start_word + width
                    if end_word >= prepared.text_len:
                        continue
                    score = self.sigmoid(float(logits[0, pos, width, label_pos]))
                    if score > best:
                        best = score
            scores.append(best)
        return scores

    def format_classification(
        self,
        scores: List[float],
        labels: Sequence[str],
        multi_label: bool,
        threshold: float,
    ) -> List[str] | str:
        pairs = sorted(zip(labels, scores), key=lambda item: -item[1])
        if multi_label:
            chosen = [label for label, score in pairs if score >= threshold]
            if not chosen and pairs:
                chosen = [pairs[0][0]]
            return chosen
        return pairs[0][0] if pairs else ""

    def classify_text(
        self,
        logits: np.ndarray,
        prepared: PreparedInputs,
        task_name: str,
        labels: Sequence[str],
        threshold: float,
        multi_label: bool,
    ) -> Dict[str, List[str] | str]:
        scores = self.classification_scores(logits, prepared, labels)
        return {
            task_name: self.format_classification(scores, labels, multi_label, threshold),
        }

    def parse_field_spec(self, spec: str) -> Tuple[str, str, Optional[str]]:
        parts = spec.split("::")
        name = parts[0]
        dtype = "list"
        description = None
        dtype_explicit = False
        for part in parts[1:]:
            if part in ("str", "list"):
                dtype = part
                dtype_explicit = True
            elif part.startswith("[") and part.endswith("]"):
                if not dtype_explicit:
                    dtype = "str"
            elif description is None:
                description = part
            else:
                description += f"::{part}"
        return name, dtype, description

    def extract_json(
        self,
        logits: np.ndarray,
        prepared: PreparedInputs,
        parent: str,
        fields: Sequence[str],
        threshold: float,
    ) -> Dict[str, List[Dict[str, Optional[str] | List[str]]]]:
        parsed = [self.parse_field_spec(spec) for spec in fields]
        labels = [name for name, _dtype, _desc in parsed]
        field_map = {name: (dtype, desc) for name, dtype, desc in parsed}

        spans_by_label: Dict[str, List[Tuple[str, float, int, int]]] = {}
        for label_index, label in enumerate(labels):
            label_pos = prepared.label_positions[label_index]
            spans_by_label[label] = self.find_spans_for_label(logits, label_pos, prepared, threshold)

        obj: Dict[str, Optional[str] | List[str]] = {}
        for label in labels:
            spans = spans_by_label[label]
            dtype, _desc = field_map[label]
            if dtype == "str":
                best = self.choose_best_span(spans)
                obj[label] = best[0] if best else None
            else:
                obj[label] = self.format_spans(spans)
        return {parent: [obj]}


def build_validation_cases() -> Tuple[EntitiesCase, ClassificationCase, JsonCase]:
    return (
        EntitiesCase(
            text="Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
            labels=["company", "person", "product", "location"],
            descriptions={"company": "Organization or business names"},
        ),
        ClassificationCase(
            text="This laptop has amazing performance but terrible battery life!",
            task_name="sentiment",
            labels=["positive", "negative", "neutral"],
        ),
        JsonCase(
            text="iPhone 15 Pro Max with 256GB storage, priced at $1199.",
            parent="product",
            fields=[
                "name::str::Full product name and model",
                "storage::str::Storage capacity",
                "price::str::Product price with currency",
            ],
        ),
    )


def compare_outputs(name: str, torch_out: object, onnx_out: object) -> None:
    if torch_out != onnx_out:
        raise ValueError(f"{name} output mismatch.\nTorch: {torch_out}\nONNX:  {onnx_out}")


def run_torch_logits(wrapper: SpanLogitsWrapper, inputs: InputTensors) -> np.ndarray:
    with torch.no_grad():
        return wrapper(*inputs).detach().cpu().numpy()


def validate_extraction_methods(
    onnx_path: Path,
    wrapper: SpanLogitsWrapper,
    decoder: SpanLogitsDecoder,
    include_token_type_ids: bool,
) -> None:
    session = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    entities_case, classification_case, json_case = build_validation_cases()

    prompt = decoder.build_prompt("entities", entities_case.descriptions)
    schema_tokens = decoder.schema_tokens_for(prompt, entities_case.labels, "[E]")
    prepared = decoder.prepare_inputs(entities_case.text, schema_tokens, len(entities_case.labels))
    inputs: InputTensors = (prepared.input_ids, prepared.attention_mask)
    if include_token_type_ids:
        inputs = (prepared.input_ids, prepared.attention_mask, torch.zeros_like(prepared.input_ids))
    torch_logits = run_torch_logits(wrapper, inputs)
    onnx_logits = run_ort(session, inputs, include_token_type_ids)
    torch_entities = decoder.extract_entities(torch_logits, prepared, entities_case.labels, entities_case.threshold)
    onnx_entities = decoder.extract_entities(onnx_logits, prepared, entities_case.labels, entities_case.threshold)
    compare_outputs("entities", torch_entities, onnx_entities)

    cls_prompt = decoder.build_prompt(classification_case.task_name, {})
    cls_tokens = decoder.schema_tokens_for(cls_prompt, classification_case.labels, "[L]")
    prepared_cls = decoder.prepare_inputs(
        classification_case.text,
        cls_tokens,
        len(classification_case.labels),
    )
    inputs = (prepared_cls.input_ids, prepared_cls.attention_mask)
    if include_token_type_ids:
        inputs = (prepared_cls.input_ids, prepared_cls.attention_mask, torch.zeros_like(prepared_cls.input_ids))
    torch_logits = run_torch_logits(wrapper, inputs)
    onnx_logits = run_ort(session, inputs, include_token_type_ids)
    torch_cls = decoder.classify_text(
        torch_logits,
        prepared_cls,
        classification_case.task_name,
        classification_case.labels,
        classification_case.threshold,
        classification_case.multi_label,
    )
    onnx_cls = decoder.classify_text(
        onnx_logits,
        prepared_cls,
        classification_case.task_name,
        classification_case.labels,
        classification_case.threshold,
        classification_case.multi_label,
    )
    compare_outputs("classification", torch_cls, onnx_cls)

    json_prompt = decoder.build_prompt(json_case.parent, {})
    json_tokens = decoder.schema_tokens_for(json_prompt, [f.split("::")[0] for f in json_case.fields], "[C]")
    prepared_json = decoder.prepare_inputs(
        json_case.text,
        json_tokens,
        len(json_case.fields),
    )
    inputs = (prepared_json.input_ids, prepared_json.attention_mask)
    if include_token_type_ids:
        inputs = (prepared_json.input_ids, prepared_json.attention_mask, torch.zeros_like(prepared_json.input_ids))
    torch_logits = run_torch_logits(wrapper, inputs)
    onnx_logits = run_ort(session, inputs, include_token_type_ids)
    torch_json = decoder.extract_json(
        torch_logits,
        prepared_json,
        json_case.parent,
        json_case.fields,
        json_case.threshold,
    )
    onnx_json = decoder.extract_json(
        onnx_logits,
        prepared_json,
        json_case.parent,
        json_case.fields,
        json_case.threshold,
    )
    compare_outputs("json", torch_json, onnx_json)


def label_token_ids_for(tokenizer: PreTrainedTokenizerBase, tokens: Sequence[str]) -> List[int]:
    ids = [tokenizer.convert_tokens_to_ids(token) for token in tokens]
    missing = [token for token, token_id in zip(tokens, ids) if token_id is None or token_id == tokenizer.unk_token_id]
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
        max_abs = validate_onnx(onnx_path, wrapper, inputs, config.include_token_type_ids)
        print(f"Validation OK (max abs diff: {max_abs:.6f})")

    if config.quantize and config.validate_quantized:
        shape = validate_quantized(
            config.output_dir / "model_int8.onnx",
            inputs,
            config.include_token_type_ids,
        )
        print(f"Quantized model loads OK (output shape: {shape})")

    if config.validate and config.validate_extraction:
        decoder = SpanLogitsDecoder(tokenizer, extractor.max_width, config.max_seq_len)
        validate_extraction_methods(
            onnx_path,
            wrapper,
            decoder,
            config.include_token_type_ids,
        )
        print("Extraction validation OK (entities, classification, json)")


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(description="Export fastino/gliner2-multi-v1 to ONNX.")
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
