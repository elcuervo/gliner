# frozen_string_literal: true

module Gliner
  PreparedInput = Data.define(
    :input_ids,
    :word_ids,
    :attention_mask,
    :words_mask,
    :pos_to_word_index,
    :start_map,
    :end_map,
    :original_text,
    :text_len
  )
end
