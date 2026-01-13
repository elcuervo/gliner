import torch
from typing import Optional

from gliner2 import GLiNER2


Tensor = torch.Tensor


class SpanLogitsWrapper(torch.nn.Module):
    def __init__(self, extractor: GLiNER2, p_token_id: int):
        super().__init__()
        self.encoder = extractor.encoder
        self.span_rep = extractor.span_rep
        self.count_embed = extractor.count_embed
        self.classifier = extractor.classifier
        self.max_width = int(extractor.max_width)
        self.register_buffer(
            "p_token_id",
            torch.tensor(int(p_token_id), dtype=torch.long),
            persistent=False,
        )

    def _text_embeddings(self, token_embeddings: Tensor, words_mask: Tensor) -> Tensor:
        # Assumes batch size 1; words_mask marks first subword per word.
        positions = torch.nonzero(words_mask[0], as_tuple=False).squeeze(-1)
        return token_embeddings[0].index_select(0, positions).unsqueeze(0)

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
    ) -> tuple[Tensor, Tensor]:
        if token_type_ids is None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        token_embeddings = outputs.last_hidden_state  # (1, seq_len, hidden)
        # Keep task_type/text_lengths as graph inputs.
        token_embeddings = token_embeddings + (
            task_type.to(token_embeddings.dtype).sum()
            + text_lengths.to(token_embeddings.dtype).sum()
        ) * 0.0

        # Prompt embedding ([P]) for count-aware projection.
        p_mask = input_ids[0].eq(self.p_token_id)
        p_pos = torch.argmax(p_mask.to(torch.long))
        p_emb = token_embeddings[0, p_pos, :]

        # Label embeddings (shape: num_labels, hidden)
        label_positions = label_positions[0].to(torch.long)
        label_embs = token_embeddings[0, label_positions, :]
        label_embs = label_embs * label_mask[0].unsqueeze(-1)

        # Classification logits (shape: num_labels)
        cls_logits = self.classifier(label_embs).squeeze(-1)

        # Span logits (shape: text_len, max_width, num_labels)
        text_embeddings = self._text_embeddings(token_embeddings, words_mask)
        span_rep = self._span_rep(text_embeddings)[0]  # (text_len, max_width, hidden)
        struct_proj = self.count_embed(label_embs, 1)  # (1, num_labels, hidden)
        span_scores = torch.sigmoid(torch.einsum("lwd,pkd->plwk", span_rep, struct_proj))
        logits = span_scores[0]  # (text_len, max_width, num_labels)

        return logits.unsqueeze(0), cls_logits.unsqueeze(0)
