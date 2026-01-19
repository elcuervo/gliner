# frozen_string_literal: true

module Gliner
  class InputBuilder
    require 'set'

    def initialize(text_processor, max_seq_len:)
      @text_processor = text_processor
      @max_seq_len = max_seq_len
    end

    def prepare(text, schema_tokens, already_normalized: false)
      normalized_text = normalize_text(text, already_normalized: already_normalized)
      words, start_map, end_map = @text_processor.split_words(normalized_text)
      input_ids, word_ids = encode_tokens(schema_tokens, words)
      input_ids, word_ids = truncate_inputs(input_ids, word_ids, max_len: @max_seq_len)

      text_start_index = schema_tokens.length + 1
      text_len = infer_effective_text_len(word_ids, text_start_index, words.length)

      context = {
        input_ids: input_ids,
        word_ids: word_ids,
        text_start_index: text_start_index,
        start_map: start_map,
        end_map: end_map,
        original_text: normalized_text,
        text_len: text_len
      }

      build_prepared_input(context)
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

    def normalize_text(text, already_normalized:)
      already_normalized ? text.to_s : @text_processor.normalize_text(text)
    end

    def encode_tokens(schema_tokens, words)
      combined_tokens = schema_tokens + ['[SEP_TEXT]'] + words
      encoded = @text_processor.encode_pretokenized(combined_tokens)
      [encoded[:ids], encoded[:word_ids]]
    end

    def truncate_inputs(input_ids, word_ids, max_len:)
      return [input_ids, word_ids] if input_ids.length <= max_len

      [input_ids.take(max_len), word_ids.take(max_len)]
    end

    def build_prepared_input(context)
      input_ids = context.fetch(:input_ids)
      word_ids = context.fetch(:word_ids)
      text_start_index = context.fetch(:text_start_index)

      word_analysis = analyze_words(word_ids, text_start_index)

      PreparedInput.new(
        input_ids: input_ids,
        word_ids: word_ids,
        attention_mask: Array.new(input_ids.length, 1),
        words_mask: word_analysis[:mask],
        pos_to_word_index: word_analysis[:index_map],
        start_map: context.fetch(:start_map),
        end_map: context.fetch(:end_map),
        original_text: context.fetch(:original_text),
        text_len: context.fetch(:text_len)
      )
    end

    def analyze_words(word_ids, text_start_index)
      mask = Array.new(word_ids.length, 0)
      index_map = Array.new(word_ids.length)
      last_word_id = nil
      seen = Set.new

      word_ids.each_with_index do |word_id, i|
        next unless word_id

        # Build mask (word boundaries)
        if word_id != last_word_id && word_id >= text_start_index
          mask[i] = 1
          last_word_id = word_id
        end

        # Build index map (first occurrence)
        unless seen.include?(word_id)
          seen << word_id
          index_map[i] = word_id - text_start_index if word_id >= text_start_index
        end
      end

      { mask: mask, index_map: index_map }
    end

    def infer_effective_text_len(word_ids, text_start_index, full_text_len)
      max_text_word_id = word_ids.compact.select { |word_id| word_id >= text_start_index }.max
      return full_text_len if max_text_word_id.nil?

      present = (max_text_word_id - text_start_index) + 1
      [present, full_text_len].min
    end
  end
end
