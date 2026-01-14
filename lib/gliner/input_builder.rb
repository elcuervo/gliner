# frozen_string_literal: true

module Gliner
  class InputBuilder
    def initialize(text_processor, max_seq_len:)
      @text_processor = text_processor
      @max_seq_len = max_seq_len
    end

    def prepare(text, schema_tokens, already_normalized: false)
      normalized_text = already_normalized ? text.to_s : @text_processor.normalize_text(text)
      words, start_map, end_map = @text_processor.split_words(normalized_text)
      combined_tokens = schema_tokens + ["[SEP_TEXT]"] + words

      encoded = @text_processor.encode_pretokenized(combined_tokens)
      input_ids = encoded[:ids]
      word_ids = encoded[:word_ids]

      truncated = truncate_inputs!(input_ids, word_ids, max_len: @max_seq_len)
      input_ids = truncated[:input_ids]
      word_ids = truncated[:word_ids]

      text_start_combined = schema_tokens.length + 1
      full_text_len = words.length
      effective_text_len = infer_effective_text_len(word_ids, text_start_combined, full_text_len)

      {
        input_ids: input_ids,
        word_ids: word_ids,
        attention_mask: Array.new(input_ids.length, 1),
        words_mask: build_words_mask(word_ids, text_start_combined),
        pos_to_word_index: build_pos_to_word_index(word_ids, text_start_combined),
        start_map: start_map,
        end_map: end_map,
        original_text: normalized_text,
        text_len: effective_text_len
      }
    end

    def schema_tokens_for(prompt:, labels:, label_prefix:)
      tokens = ["(", "[P]", prompt.to_s, "("]

      labels.each do |label|
        tokens << label_prefix
        tokens << label.to_s
      end

      tokens.concat([")", ")"])
      tokens
    end

    private

    def truncate_inputs!(input_ids, word_ids, max_len:)
      return { input_ids: input_ids, word_ids: word_ids } if input_ids.length <= max_len
      { input_ids: input_ids.take(max_len), word_ids: word_ids.take(max_len) }
    end

    def build_words_mask(word_ids, text_start_combined)
      mask = Array.new(word_ids.length, 0)
      last_wid = nil

      word_ids.each_with_index do |wid, i|
        next if wid.nil?
        if wid != last_wid
          mask[i] = 1 if wid >= text_start_combined
          last_wid = wid
        end
      end
      mask
    end

    def build_pos_to_word_index(word_ids, text_start_combined)
      map = Array.new(word_ids.length)
      seen = {}
      word_ids.each_with_index do |wid, i|
        next if wid.nil?
        next if seen.key?(wid)
        seen[wid] = true
        map[i] = wid - text_start_combined if wid >= text_start_combined
      end
      map
    end

    def infer_effective_text_len(word_ids, text_start_combined, full_text_len)
      max_text_wid = word_ids.compact.select { |wid| wid >= text_start_combined }.max
      return full_text_len if max_text_wid.nil?

      present = (max_text_wid - text_start_combined) + 1
      [present, full_text_len].min
    end
  end
end
