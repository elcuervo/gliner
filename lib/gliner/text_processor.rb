# frozen_string_literal: true

module Gliner
  class TextProcessor
    def initialize(tokenizer)
      @tokenizer = tokenizer
      @word_pre_tokenizer = Tokenizers::PreTokenizers::BertPreTokenizer.new
    end

    def normalize_text(text)
      str = text.to_s
      str = "." if str.empty?
      str.end_with?(".", "!", "?") ? str : "#{str}."
    end

    def split_words(text)
      text = text.to_s
      tokens = []
      starts = []
      ends = []

      @word_pre_tokenizer.pre_tokenize_str(text).each do |(token, (start_pos, end_pos))|
        token = token.to_s.downcase

        next if token.empty?

        tokens << token
        starts << start_pos
        ends << end_pos
      end

      [tokens, starts, ends]
    end

    def encode_pretokenized(tokens)
      enc = @tokenizer.encode(tokens, is_pretokenized: true, add_special_tokens: false)

      { ids: enc.ids, word_ids: enc.word_ids }
    end
  end
end
