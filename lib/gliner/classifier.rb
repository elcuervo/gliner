# frozen_string_literal: true

require 'gliner/position_iteration'

module Gliner
  class Classifier
    include PositionIteration

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
      seq_len = logits[0].length

      scores = each_position_width(seq_len, prepared, @max_width).map do |pos, _start_word, width|
        logit = @inference.label_logit(logits, pos, width, label_index, label_positions)
        @inference.sigmoid(logit)
      end

      scores.max || -Float::INFINITY
    end

    def sorted_label_scores(scores, labels)
      scores
        .each_with_index.map { |score, i| [labels.fetch(i), score] }
        .sort_by { |(_label, score)| -score }
    end

    def format_multi_label(label_scores, cls_threshold, include_confidence)
      chosen = labels_above_threshold(label_scores, cls_threshold)

      chosen.map { |label, score| format_label(label, score, include_confidence) }
    end

    def labels_above_threshold(label_scores, threshold)
      above = label_scores.select { |_label, score| score >= threshold }
      above.empty? && label_scores.first ? [label_scores.first] : above
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
