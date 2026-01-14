# frozen_string_literal: true

require_relative 'prepared_input'

module Gliner
  class InputBuilder
    def initialize(text_processor, max_seq_len:)
      @text_processor = text_processor
      @max_seq_len = max_seq_len
    end

    def prepare(text, schema_tokens, already_normalized: false)
      normalized_text = already_normalized ? text.to_s : @text_processor.normalize_text(text)
      words, start_map, end_map = @text_processor.split_words(normalized_text)
      combined_tokens = schema_tokens + ['[SEP_TEXT]'] + words

      encoded = @text_processor.encode_pretokenized(combined_tokens)
      input_ids = encoded[:ids]
      word_ids = encoded[:word_ids]

      truncated = truncate_inputs(input_ids, word_ids, max_len: @max_seq_len)
      input_ids = truncated[:input_ids]
      word_ids = truncated[:word_ids]

      text_start_index = schema_tokens.length + 1
      full_text_len = words.length
      effective_text_len = infer_effective_text_len(word_ids, text_start_index, full_text_len)

      PreparedInput.new(
        input_ids: input_ids,
        word_ids: word_ids,
        attention_mask: Array.new(input_ids.length, 1),
        words_mask: build_words_mask(word_ids, text_start_index),
        pos_to_word_index: build_pos_to_word_index(word_ids, text_start_index),
        start_map: start_map,
        end_map: end_map,
        original_text: normalized_text,
        text_len: effective_text_len
      )
    end

    def schema_tokens_for(prompt:, labels:, label_prefix:)
      tokens = ['(', '[P]', prompt.to_s, '(']

      labels.each do |label|
        tokens << label_prefix
        tokens << label.to_s
      end

      tokens.push(')', ')')
      tokens
    end

    private

    def truncate_inputs(input_ids, word_ids, max_len:)
      return { input_ids: input_ids, word_ids: word_ids } if input_ids.length <= max_len

      { input_ids: input_ids.take(max_len), word_ids: word_ids.take(max_len) }
    end

    def build_words_mask(word_ids, text_start_index)
      mask = Array.new(word_ids.length, 0)
      last_word_id = nil

      word_ids.each_with_index do |word_id, i|
        next if word_id.nil?

        if word_id != last_word_id
          mask[i] = 1 if word_id >= text_start_index
          last_word_id = word_id
        end
      end
      mask
    end

    def build_pos_to_word_index(word_ids, text_start_index)
      index_map = Array.new(word_ids.length)
      seen = {}
      word_ids.each_with_index do |word_id, i|
        next if word_id.nil?
        next if seen.key?(word_id)

        seen[word_id] = true
        index_map[i] = word_id - text_start_index if word_id >= text_start_index
      end
      index_map
    end

    def infer_effective_text_len(word_ids, text_start_index, full_text_len)
      max_text_word_id = word_ids.compact.select { |word_id| word_id >= text_start_index }.max
      return full_text_len if max_text_word_id.nil?

      present = (max_text_word_id - text_start_index) + 1
      [present, full_text_len].min
    end
  end
end
