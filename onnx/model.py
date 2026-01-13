import torch
from typing import Optional, Sequence

from gliner2 import GLiNER2


Tensor = torch.Tensor


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
