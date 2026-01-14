# frozen_string_literal: true

module Gliner
  class Classifier
    def initialize(inference, max_width:)
      @inference = inference
      @max_width = max_width
    end

    def classification_scores(logits:, labels:, label_positions:, pos_to_word_index:, text_len:, threshold:)
      scores = []

      labels.each_index do |label_index|
        max = -Float::INFINITY
        seq_len = logits[0].length
        (0...seq_len).each do |pos|
          start_word = pos_to_word_index[pos]
          next if start_word.nil?

          (0...@max_width).each do |width|
            end_word = start_word + width
            next if end_word >= text_len
            s = @inference.sigmoid(@inference.label_logit(logits, pos, width, label_index, label_positions))
            max = s if s > max
          end
        end
        scores << max
      end

      scores
    end

    def format_classification(scores, labels:, multi_label:, include_confidence:, cls_threshold:)
      pairs = scores.each_with_index.map { |s, i| [labels.fetch(i), s] }
      pairs.sort_by! { |(_i, s)| -s }

      if multi_label
        chosen = pairs.select { |(_label, s)| s >= cls_threshold }
        chosen = [pairs.first] if chosen.empty? && pairs.first
        chosen.map! { |(label, s)| include_confidence ? { "label" => label, "confidence" => s } : label }
      else
        label, s = pairs.first
        include_confidence ? { "label" => label, "confidence" => s } : label
      end
    end
  end
end
