# frozen_string_literal: true

module Gliner
  class Inference
    Request = Data.define(
      :input_ids,
      :attention_mask,
      :words_mask,
      :text_lengths,
      :task_type,
      :label_positions,
      :label_mask,
      :want_cls
    )
  end
end
