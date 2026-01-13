from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import PreTrainedTokenizerBase


Tensor = torch.Tensor


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
