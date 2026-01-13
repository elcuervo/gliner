import torch
from typing import Optional

from gliner2 import GLiNER2


Tensor = torch.Tensor


class SpanLogitsWrapper(torch.nn.Module):
    def __init__(self, extractor: GLiNER2):
        super().__init__()
        self.encoder = extractor.encoder
        self.span_rep = extractor.span_rep
        self.count_embed = extractor.count_embed
        self.classifier = extractor.classifier
        self.max_width = int(extractor.max_width)

    def _text_embeddings(self, token_embeddings: Tensor, words_mask: Tensor) -> Tensor:
        # Assumes batch size 1; words_mask marks first subword per word.
        positions = torch.nonzero(words_mask[0], as_tuple=False).squeeze(-1)
        return token_embeddings[0].index_select(0, positions).unsqueeze(0), positions

    def _span_rep(self, text_embeddings: Tensor) -> Tensor:
        seq_len = text_embeddings.shape[1]
        device = text_embeddings.device

        start = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, self.max_width)
        widths = torch.arange(self.max_width, device=device).unsqueeze(0).expand(seq_len, self.max_width)
        end = start + widths
        valid = end < seq_len
        start = torch.where(valid, start, torch.zeros_like(start))
        end = torch.where(valid, end, torch.zeros_like(end))

        spans_idx = torch.stack([start, end], dim=-1)
        spans_idx = spans_idx.view(1, seq_len * self.max_width, 2)
        span_mask = (spans_idx[:, :, 0] == -1) | (spans_idx[:, :, 1] == -1)
        safe_spans = torch.where(span_mask.unsqueeze(-1), torch.zeros_like(spans_idx), spans_idx)
        return self.span_rep(text_embeddings, safe_spans)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        words_mask: Tensor,
        text_lengths: Tensor,
        task_type: Tensor,
        label_positions: Tensor,
        label_mask: Tensor,
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
        # Keep task_type/text_lengths as graph inputs.
        token_embeddings = token_embeddings + (
            task_type.to(token_embeddings.dtype).sum()
            + text_lengths.to(token_embeddings.dtype).sum()
        ) * 0.0

        seq_len = input_ids.shape[1]
        num_labels = label_positions.shape[1]
        device = token_embeddings.device

        # Label embeddings (shape: num_labels, hidden)
        label_positions = label_positions[0].to(torch.long)
        label_embs = token_embeddings[0, label_positions, :]
        label_embs = label_embs * label_mask[0].unsqueeze(-1)

        # Classification logits (shape: num_labels)
        cls_logits = self.classifier(label_embs).squeeze(-1)

        # Span logits for text tokens only (shape: text_len, max_width, num_labels)
        text_embeddings, word_positions = self._text_embeddings(token_embeddings, words_mask)
        span_rep = self._span_rep(text_embeddings)[0]
        struct_proj = self.count_embed(label_embs, 1)
        span_logits = torch.einsum("lwd,pkd->plwk", span_rep, struct_proj)[0]

        # Scatter span logits back into full sequence length.
        fill_value = -1000.0
        logits_span = torch.full(
            (1, seq_len, self.max_width, num_labels), fill_value, device=device
        )
        index = word_positions.view(1, -1, 1, 1).expand_as(span_logits.unsqueeze(0))
        logits_span = logits_span.scatter(1, index, span_logits.unsqueeze(0))

        # Build classification logits tensor at a single text position.
        logits_cls = torch.full(
            (1, seq_len, self.max_width, num_labels), fill_value, device=device
        )
        safe_positions = torch.cat(
            [word_positions, torch.zeros(1, device=device, dtype=word_positions.dtype)]
        )
        cls_pos = safe_positions[0]
        logits_cls[0, cls_pos, 0, :] = cls_logits

        # Select between span and classification logits based on task_type.
        cls_mask = (task_type == 1).to(token_embeddings.dtype).view(-1, 1, 1, 1)
        logits = logits_span * (1 - cls_mask) + logits_cls * cls_mask

        return logits
