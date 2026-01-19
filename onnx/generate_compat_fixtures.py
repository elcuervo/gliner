import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from decoder import SpanLogitsDecoder
from export import ExportConfig, export


def load_config(model_dir: str) -> Dict[str, int]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return {"max_width": 8, "max_seq_len": 512}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_inputs(prepared, task_type_val: int, num_labels: int, include_token_type_ids: bool):
    words_mask = torch.tensor(
        [1 if idx is not None else 0 for idx in prepared.pos_to_word_index],
        dtype=torch.long,
    ).unsqueeze(0)
    text_lengths = torch.tensor([prepared.text_len], dtype=torch.long)
    task_type = torch.tensor([task_type_val], dtype=torch.long)
    label_positions = torch.tensor([prepared.label_positions], dtype=torch.long)
    label_mask = torch.ones((1, num_labels), dtype=torch.long)
    inputs = (
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


def run_ort(session: ort.InferenceSession, inputs, include_token_type_ids: bool):
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
    outputs = session.run(["logits"], feed)
    return outputs[0]


def generate_entities_cases(n: int) -> List[Dict]:
    companies = ["Apple", "Google", "Microsoft", "Amazon", "Meta", "Tesla", "Nike", "Samsung", "Intel", "Sony"]
    people = ["Tim Cook", "Sundar Pichai", "Satya Nadella", "Andy Jassy", "Mark Zuckerberg", "Elon Musk", "Phil Knight", "Jong-Hee Han", "Pat Gelsinger", "Kenichiro Yoshida"]
    products = ["iPhone 15", "Pixel 9", "Surface Pro", "Echo Show", "Quest 3", "Model Y", "Air Max", "Galaxy S24", "Core i9", "PlayStation 5"]
    locations = ["Cupertino", "Mountain View", "Redmond", "Seattle", "Menlo Park", "Austin", "Portland", "Seoul", "Santa Clara", "Tokyo"]
    templates = [
        "{company} CEO {person} announced {product} in {location}.",
        "At {location}, {company} launched the new {product}, said {person}.",
        "{person} from {company} showcased {product} during a keynote in {location}.",
        "{company} unveiled {product} today in {location} with {person} on stage.",
        "{location} hosted the {company} event where {person} introduced {product}.",
    ]
    descriptions = {"company": "Organization or business names"}
    labels = ["company", "person", "product", "location"]

    cases = []
    for idx in range(n):
        text = templates[idx % len(templates)].format(
            company=companies[idx % len(companies)],
            person=people[(idx * 3) % len(people)],
            product=products[(idx * 5) % len(products)],
            location=locations[(idx * 7) % len(locations)],
        )
        cases.append({
            "text": text,
            "labels": labels,
            "descriptions": descriptions,
            "threshold": 0.5,
        })
    return cases


def generate_classification_cases(n: int) -> List[Dict]:
    sentiment_texts = [
        "This phone is amazing and fast, I love it.",
        "The battery life is terrible and disappointing.",
        "It's okay, nothing special but not bad either.",
        "Outstanding performance with great screen quality.",
        "The device is slow and frustrating to use.",
    ]
    aspect_texts = [
        "Great camera and screen but the battery is weak.",
        "Affordable price with solid performance.",
        "Battery and camera are excellent, price is high.",
        "The screen is dim but the battery lasts long.",
        "Love the camera, hate the price.",
    ]

    cases = []
    for idx in range(n):
        if idx % 2 == 0:
            text = sentiment_texts[idx % len(sentiment_texts)]
            cases.append({
                "text": text,
                "task_name": "sentiment",
                "labels": ["positive", "negative", "neutral"],
                "threshold": 0.5,
                "multi_label": False,
            })
        else:
            text = aspect_texts[idx % len(aspect_texts)]
            cases.append({
                "text": text,
                "task_name": "aspects",
                "labels": ["camera", "battery", "screen", "price", "performance"],
                "threshold": 0.4,
                "multi_label": True,
            })
    return cases


def generate_json_cases(n: int) -> List[Dict]:
    names = ["iPhone 15 Pro", "Galaxy S24", "Pixel 9 Pro", "Xperia 5", "OnePlus 12", "ThinkPad X1", "MacBook Pro", "Surface Laptop", "iPad Air", "Kindle Scribe"]
    storages = ["128GB", "256GB", "512GB", "1TB", "64GB"]
    prices = ["$799", "$999", "$1199", "$1499", "$699", "$899"]
    fields = [
        "name::str::Full product name",
        "storage::str::Storage capacity",
        "price::str::Product price",
    ]

    cases = []
    for idx in range(n):
        text = "Product: {name}. Storage: {storage}. Price: {price}.".format(
            name=names[idx % len(names)],
            storage=storages[(idx * 2) % len(storages)],
            price=prices[(idx * 3) % len(prices)],
        )
        cases.append({
            "text": text,
            "parent": "product",
            "fields": fields,
            "threshold": 0.4,
        })
    return cases


def ensure_onnx_export(repo_id: str, output_dir: str, model_file: str) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / model_file
    if model_path.exists():
        return output_path.as_posix()

    quantize = model_file == "model_int8.onnx"
    export(
        ExportConfig(
            model_id=repo_id,
            output_dir=output_path,
            max_seq_len=512,
            opset=19,
            include_token_type_ids=False,
            quantize=quantize,
            validate=False,
            validate_quantized=False,
            validate_extraction=False,
        )
    )

    return output_path.as_posix()


def main() -> None:
    random.seed(42)
    repo_id = os.environ.get("GLINER_REPO_ID", "fastino/gliner2-multi-v1")
    model_file = os.environ.get("GLINER_MODEL_FILE", "model.onnx")
    onnx_dir = os.environ.get(
        "GLINER_ONNX_DIR",
        os.path.join(".context", "onnx", repo_id.replace("/", "__")),
    )
    model_dir = ensure_onnx_export(repo_id, onnx_dir, model_file)
    model_path = os.path.join(model_dir, model_file)

    config = load_config(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    decoder = SpanLogitsDecoder(
        tokenizer,
        max_width=config.get("max_width", 8),
        max_seq_len=config.get("max_seq_len", 512),
    )

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    include_token_type_ids = "token_type_ids" in input_names

    entities_cases = generate_entities_cases(100)
    classification_cases = generate_classification_cases(100)
    json_cases = generate_json_cases(100)

    fixtures = {
        "meta": {
            "repo_id": repo_id,
            "model_dir": model_dir,
            "model_file": model_file,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "entities": [],
        "classification": [],
        "json": [],
    }

    for case in entities_cases:
        prompt = decoder.build_prompt("entities", case["descriptions"])
        schema_tokens = decoder.schema_tokens_for(prompt, case["labels"], "[E]")
        prepared = decoder.prepare_inputs(case["text"], schema_tokens, len(case["labels"]))
        inputs = build_inputs(prepared, task_type_val=0, num_labels=len(case["labels"]), include_token_type_ids=include_token_type_ids)
        logits = run_ort(session, inputs, include_token_type_ids)
        expected = decoder.extract_entities(logits, prepared, case["labels"], case["threshold"])
        fixtures["entities"].append({"input": case, "expected": expected})

    for case in classification_cases:
        prompt = decoder.build_prompt(case["task_name"], {})
        schema_tokens = decoder.schema_tokens_for(prompt, case["labels"], "[L]")
        prepared = decoder.prepare_inputs(case["text"], schema_tokens, len(case["labels"]))
        inputs = build_inputs(prepared, task_type_val=1, num_labels=len(case["labels"]), include_token_type_ids=include_token_type_ids)
        logits = run_ort(session, inputs, include_token_type_ids)
        expected = decoder.classify_text(
            logits,
            prepared,
            case["task_name"],
            case["labels"],
            case["threshold"],
            case["multi_label"],
        )
        fixtures["classification"].append({"input": case, "expected": expected})

    for case in json_cases:
        prompt = decoder.build_prompt(case["parent"], {})
        labels = [field.split("::")[0] for field in case["fields"]]
        schema_tokens = decoder.schema_tokens_for(prompt, labels, "[C]")
        prepared = decoder.prepare_inputs(case["text"], schema_tokens, len(case["fields"]))
        inputs = build_inputs(prepared, task_type_val=2, num_labels=len(case["fields"]), include_token_type_ids=include_token_type_ids)
        logits = run_ort(session, inputs, include_token_type_ids)
        expected = decoder.extract_json(
            logits,
            prepared,
            case["parent"],
            case["fields"],
            case["threshold"],
        )
        fixtures["json"].append({"input": case, "expected": expected})

    os.makedirs(os.path.join("spec", "fixtures"), exist_ok=True)
    output_path = os.path.join("spec", "fixtures", "python_compat.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2, ensure_ascii=True)

    print(f"Wrote fixtures to {output_path}")


if __name__ == "__main__":
    main()
