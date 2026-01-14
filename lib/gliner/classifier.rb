# frozen_string_literal: true

require_relative 'prepared_input'

module Gliner
  class Classifier
    def initialize(inference, max_width:)
      @inference = inference
      @max_width = max_width
    end

    def classification_scores(logits, labels, label_positions, prepared)
      scores = []

      labels.each_index do |label_index|
        max_score = -Float::INFINITY
        seq_len = logits[0].length

        (0...seq_len).each do |pos|
          start_word = prepared.pos_to_word_index[pos]
          next if start_word.nil?

          (0...@max_width).each do |width|
            end_word = start_word + width
            next if end_word >= prepared.text_len

            score = @inference.sigmoid(@inference.label_logit(logits, pos, width, label_index, label_positions))
            max_score = score if score > max_score
          end
        end
        scores << max_score
      end

      scores
    end

    def format_classification(scores, labels:, multi_label:, include_confidence:, cls_threshold:)
      label_scores = scores.each_with_index.map { |score, i| [labels.fetch(i), score] }
      label_scores.sort_by! { |(_label, score)| -score }

      if multi_label
        chosen = label_scores.select { |(_label, score)| score >= cls_threshold }
        chosen = [label_scores.first] if chosen.empty? && label_scores.first
        chosen.map! do |(label, score)|
          include_confidence ? { 'label' => label, 'confidence' => score } : label
        end
      else
        label, score = label_scores.first
        include_confidence ? { 'label' => label, 'confidence' => score } : label
      end
    end
  end
end
