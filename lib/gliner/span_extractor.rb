# frozen_string_literal: true

require 'gliner/position_iteration'

module Gliner
  class SpanExtractor
    include PositionIteration

    SCORE_SIMILARITY_THRESHOLD = 0.02

    def initialize(inference, max_width:)
      @inference = inference
      @max_width = max_width
    end

    def extract_spans_by_label(logits, labels, label_positions, prepared, threshold: 0.5, thresholds_by_label: nil)
      labels.each_with_index.with_object({}) do |(label, label_index), out|
        out[label.to_s] = find_spans_for_label(
          logits: logits,
          label_index: label_index,
          label_positions: label_positions,
          prepared: prepared,
          threshold: threshold_for(label, threshold, thresholds_by_label)
        )
      end
    end

    def find_spans_for_label(logits:, label_index:, label_positions:, prepared:, threshold:)
      seq_len = logits.first.length

      each_position_width(seq_len, prepared, @max_width).filter_map do |pos, start_word, width|
        score = calculate_span_score(logits, pos, width, label_index, label_positions)
        next if score < threshold

        build_span(prepared, start_word, start_word + width, score)
      end
    end

    def choose_best_span(spans)
      return nil if spans.empty?

      sorted = spans.sort_by { |s| [-s.score, (s.end - s.start), s.text.length] }
      best = sorted.first
      best_score = best.score
      near = spans_within_threshold(sorted, best_score)

      near.min_by { |s| [(s.end - s.start), -s.score, s.text.length] } || best
    end

    def format_single_span(span, opts = nil)
      label = extract_label(opts)
      format_span(span, opts, label: label, index: 0)
    end

    def format_spans(spans, opts = nil)
      label = extract_label(opts)
      return [] if spans.empty?

      sorted = spans.sort_by { |s| -s.score }
      selected = []

      sorted.each do |span|
        overlaps = selected.any? { |s| span.overlaps?(s) }
        next if overlaps

        selected << span
      end

      selected.each_with_index.map do |span, index|
        format_span(span, opts, label: label, index: index)
      end
    end

    private

    def calculate_span_score(logits, pos, width, label_index, label_positions)
      logit = @inference.label_logit(logits, pos, width, label_index, label_positions)
      @inference.sigmoid(logit)
    end

    def spans_within_threshold(sorted_spans, best_score)
      sorted_spans.take_while { |span| (best_score - span.score) <= SCORE_SIMILARITY_THRESHOLD }
    end

    def threshold_for(label, default_threshold, thresholds_by_label)
      return default_threshold unless thresholds_by_label&.key?(label.to_s)

      thresholds_by_label.fetch(label.to_s)
    end

    def build_span(prepared, start_word, end_word, score)
      char_start = prepared.start_map[start_word]
      char_end = prepared.end_map[end_word]

      return nil if char_start.nil? || char_end.nil?

      text_span = prepared.original_text[char_start...char_end].to_s.strip

      return nil if text_span.empty?

      Span.new(text: text_span, score: score, start: char_start, end: char_end)
    end

    def format_span(span, _opts, label:, index:)
      return nil if span.nil?

      Gliner::Entity.new(
        index: index,
        offsets: [span.start, span.end],
        text: span.text,
        name: label&.to_s,
        confidence: span.score * 100.0
      )
    end

    def extract_label(opts)
      return nil unless opts.is_a?(Hash)

      opts[:label] || opts['label']
    end
  end
end
