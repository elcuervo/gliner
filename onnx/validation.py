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
) -> Tuple[np.ndarray, np.ndarray]:
    input_ids, attention_mask = inputs[0], inputs[1]
    words_mask, text_lengths, task_type, label_positions, label_mask = inputs[2:7]
    feed = {
        "input_ids": input_ids.cpu().numpy(),
        "attention_mask": attention_mask.cpu().numpy(),
        "words_mask": words_mask.cpu().numpy(),
        "text_lengths": text_lengths.cpu().numpy(),
        "task_type": task_type.cpu().numpy(),
        "label_positions": label_positions.cpu().numpy(),
        "label_mask": label_mask.cpu().numpy(),
    }
    if include_token_type_ids:
        feed["token_type_ids"] = inputs[7].cpu().numpy()
    outputs = session.run(["logits", "cls_logits"], feed)
    return outputs[0], outputs[1]


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
        torch_logits, torch_cls = wrapper(*inputs)
        torch_logits = torch_logits.detach().cpu().numpy()
        torch_cls = torch_cls.detach().cpu().numpy()
    ort_logits, ort_cls = run_ort(session, inputs, include_token_type_ids)
    if torch_logits.shape != ort_logits.shape:
        raise ValueError(
            f"ONNX logits shape mismatch: torch {torch_logits.shape} vs onnx {ort_logits.shape}"
        )
    if torch_cls.shape != ort_cls.shape:
        raise ValueError(
            f"ONNX cls_logits shape mismatch: torch {torch_cls.shape} vs onnx {ort_cls.shape}"
        )
    max_abs = max(
        float(np.max(np.abs(torch_logits - ort_logits))),
        float(np.max(np.abs(torch_cls - ort_cls))),
    )
    return max_abs


def validate_quantized(
    onnx_path: str,
    inputs: InputTensors,
    include_token_type_ids: bool,
) -> Tuple[int, ...]:
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    ort_logits, _ort_cls = run_ort(session, inputs, include_token_type_ids)
    return ort_logits.shape


def run_torch_logits(
    wrapper: SpanLogitsWrapper, inputs: InputTensors
) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        logits, cls_logits = wrapper(*inputs)
        return logits.detach().cpu().numpy(), cls_logits.detach().cpu().numpy()


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

    def build_inputs(prepared, task_type_val: int, num_labels: int) -> InputTensors:
        words_mask = torch.tensor(
            [1 if idx is not None else 0 for idx in prepared.pos_to_word_index],
            dtype=torch.long,
        ).unsqueeze(0)
        text_lengths = torch.tensor([prepared.text_len], dtype=torch.long)
        task_type = torch.tensor([task_type_val], dtype=torch.long)
        label_positions = torch.tensor([prepared.label_positions], dtype=torch.long)
        label_mask = torch.ones((1, num_labels), dtype=torch.long)
        inputs: InputTensors = (
            prepared.input_ids,
            prepared.attention_mask,
            words_mask,
            text_lengths,
            task_type,
            label_positions,
            label_mask,
        )
        if include_token_type_ids:
            inputs = inputs + (torch.zeros_like(prepared.input_ids),)
        return inputs

    # Entities
    prompt = decoder.build_prompt("entities", entities_case.descriptions)
    schema_tokens = decoder.schema_tokens_for(prompt, entities_case.labels, "[E]")
    prepared = decoder.prepare_inputs(
        entities_case.text, schema_tokens, len(entities_case.labels)
    )
    inputs = build_inputs(prepared, task_type_val=0, num_labels=len(entities_case.labels))
    torch_logits, _torch_cls = run_torch_logits(wrapper, inputs)
    onnx_logits, _onnx_cls = run_ort(session, inputs, include_token_type_ids)
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
    inputs = build_inputs(
        prepared_cls, task_type_val=1, num_labels=len(classification_case.labels)
    )
    _torch_logits, torch_cls_logits = run_torch_logits(wrapper, inputs)
    _onnx_logits, onnx_cls_logits = run_ort(session, inputs, include_token_type_ids)
    torch_cls = {
        classification_case.task_name: decoder.format_classification(
            torch.softmax(torch.tensor(torch_cls_logits[0]), dim=-1).tolist(),
            classification_case.labels,
            classification_case.multi_label,
            classification_case.threshold,
        )
    }
    onnx_cls = {
        classification_case.task_name: decoder.format_classification(
            torch.softmax(torch.tensor(onnx_cls_logits[0]), dim=-1).tolist(),
            classification_case.labels,
            classification_case.multi_label,
            classification_case.threshold,
        )
    }
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
    inputs = build_inputs(prepared_json, task_type_val=2, num_labels=len(json_case.fields))
    torch_logits, _torch_cls = run_torch_logits(wrapper, inputs)
    onnx_logits, _onnx_cls = run_ort(session, inputs, include_token_type_ids)
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
