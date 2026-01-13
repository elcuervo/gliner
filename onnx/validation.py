from typing import Tuple

import numpy as np
import onnxruntime as ort
import torch

from decoder import SpanLogitsDecoder
from model import SpanLogitsWrapper
from validation_cases import build_validation_cases


Tensor = torch.Tensor
InputTensors = Tuple[Tensor, ...]


def run_ort(
    session: ort.InferenceSession, inputs: InputTensors, include_token_type_ids: bool
) -> np.ndarray:
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
    onnx_path: str,
    wrapper: SpanLogitsWrapper,
    inputs: InputTensors,
    include_token_type_ids: bool,
) -> float:
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    with torch.no_grad():
        torch_out = wrapper(*inputs).detach().cpu().numpy()
    ort_out = run_ort(session, inputs, include_token_type_ids)
    if torch_out.shape != ort_out.shape:
        raise ValueError(
            f"ONNX output shape mismatch: torch {torch_out.shape} vs onnx {ort_out.shape}"
        )
    return float(np.max(np.abs(torch_out - ort_out)))


def validate_quantized(
    onnx_path: str,
    inputs: InputTensors,
    include_token_type_ids: bool,
) -> Tuple[int, ...]:
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    ort_out = run_ort(session, inputs, include_token_type_ids)
    return ort_out.shape


def run_torch_logits(wrapper: SpanLogitsWrapper, inputs: InputTensors) -> np.ndarray:
    with torch.no_grad():
        return wrapper(*inputs).detach().cpu().numpy()


def compare_outputs(name: str, torch_out: object, onnx_out: object) -> None:
    if torch_out != onnx_out:
        raise ValueError(
            f"{name} output mismatch.\nTorch: {torch_out}\nONNX:  {onnx_out}"
        )


def validate_extraction_methods(
    onnx_path: str,
    wrapper: SpanLogitsWrapper,
    decoder: SpanLogitsDecoder,
    include_token_type_ids: bool,
) -> None:
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    entities_case, classification_case, json_case = build_validation_cases()

    # Entities
    prompt = decoder.build_prompt("entities", entities_case.descriptions)
    schema_tokens = decoder.schema_tokens_for(prompt, entities_case.labels, "[E]")
    prepared = decoder.prepare_inputs(
        entities_case.text, schema_tokens, len(entities_case.labels)
    )
    inputs: InputTensors = (prepared.input_ids, prepared.attention_mask)
    if include_token_type_ids:
        inputs = (
            prepared.input_ids,
            prepared.attention_mask,
            torch.zeros_like(prepared.input_ids),
        )
    torch_logits = run_torch_logits(wrapper, inputs)
    onnx_logits = run_ort(session, inputs, include_token_type_ids)
    torch_entities = decoder.extract_entities(
        torch_logits, prepared, entities_case.labels, entities_case.threshold
    )
    onnx_entities = decoder.extract_entities(
        onnx_logits, prepared, entities_case.labels, entities_case.threshold
    )
    compare_outputs("entities", torch_entities, onnx_entities)

    # Classification
    cls_prompt = decoder.build_prompt(classification_case.task_name, {})
    cls_tokens = decoder.schema_tokens_for(
        cls_prompt, classification_case.labels, "[L]"
    )
    prepared_cls = decoder.prepare_inputs(
        classification_case.text,
        cls_tokens,
        len(classification_case.labels),
    )
    inputs = (prepared_cls.input_ids, prepared_cls.attention_mask)
    if include_token_type_ids:
        inputs = (
            prepared_cls.input_ids,
            prepared_cls.attention_mask,
            torch.zeros_like(prepared_cls.input_ids),
        )
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

    # Structured extraction
    json_prompt = decoder.build_prompt(json_case.parent, {})
    json_tokens = decoder.schema_tokens_for(
        json_prompt, [f.split("::")[0] for f in json_case.fields], "[C]"
    )
    prepared_json = decoder.prepare_inputs(
        json_case.text,
        json_tokens,
        len(json_case.fields),
    )
    inputs = (prepared_json.input_ids, prepared_json.attention_mask)
    if include_token_type_ids:
        inputs = (
            prepared_json.input_ids,
            prepared_json.attention_mask,
            torch.zeros_like(prepared_json.input_ids),
        )
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
