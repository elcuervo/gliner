# frozen_string_literal: true

module Gliner
  class Classifier
    def initialize(inference, max_width:)
      @inference = inference
      @max_width = max_width
    end

    def classification_scores(logits, labels, label_positions, prepared)
      labels.each_index.map do |label_index|
        max_label_score(logits, label_index, label_positions, prepared)
      end
    end

    def format_classification(scores, labels:, multi_label:, include_confidence:, cls_threshold:)
      label_scores = sorted_label_scores(scores, labels)

      return format_multi_label(label_scores, cls_threshold, include_confidence) if multi_label

      format_single_label(label_scores.first, include_confidence)
    end

    private

    def max_label_score(logits, label_index, label_positions, prepared)
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

      max_score
    end

    def sorted_label_scores(scores, labels)
      scores
        .each_with_index.map { |score, i| [labels.fetch(i), score] }
        .sort_by { |(_label, score)| -score }
    end

    def format_multi_label(label_scores, cls_threshold, include_confidence)
      chosen = label_scores.select { |(_label, score)| score >= cls_threshold }
      chosen = [label_scores.first] if chosen.empty? && label_scores.first

      chosen.map { |label, score| format_label(label, score, include_confidence) }
    end

    def format_single_label(label_score, include_confidence)
      label, score = label_score

      format_label(label, score, include_confidence)
    end

    def format_label(label, score, include_confidence)
      include_confidence ? { 'label' => label, 'confidence' => score } : label
    end
  end
end
