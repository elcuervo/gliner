# frozen_string_literal: true

module Gliner
  module PositionIteration
    def each_position_width(seq_len, prepared, max_width)
      return enum_for(:each_position_width, seq_len, prepared, max_width) unless block_given?

      (0...seq_len).each do |pos|
        start_word = prepared.pos_to_word_index[pos]
        next unless start_word

        (0...max_width).each do |width|
          end_word = start_word + width
          next if end_word >= prepared.text_len

          yield pos, start_word, width
        end
      end
    end
  end
end
