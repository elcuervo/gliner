# frozen_string_literal: true

module Gliner
  class SpanExtractor
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
      spans = []

      seq_len = logits[0].length
      (0...seq_len).each do |pos|
        start_word = prepared.pos_to_word_index[pos]
        next if start_word.nil?

        (0...@max_width).each do |width|
          end_word = start_word + width
          next if end_word >= prepared.text_len

          score = @inference.sigmoid(@inference.label_logit(logits, pos, width, label_index, label_positions))
          next if score < threshold

          span = build_span(prepared, start_word, end_word, score)
          spans << span if span
        end
      end

      spans
    end

    def choose_best_span(spans)
      return nil if spans.empty?

      sorted = spans.sort_by { |s| [-s.score, (s.end - s.start), s.text.length] }
      best = sorted[0]
      best_score = best.score
      near = sorted.take_while { |s| (best_score - s.score) <= SCORE_SIMILARITY_THRESHOLD }
      near.min_by { |s| [(s.end - s.start), -s.score, s.text.length] } || best
    end

    def format_single_span(span, include_confidence:, include_spans:)
      format_span(span, include_confidence: include_confidence, include_spans: include_spans)
    end

    def format_spans(spans, include_confidence:, include_spans:)
      return [] if spans.empty?

      sorted = spans.sort_by { |s| -s.score }
      selected = []

      sorted.each do |span|
        overlaps = selected.any? { |s| span.overlaps?(s) }
        next if overlaps

        selected << span
      end

      selected.map do |span|
        format_span(span, include_confidence: include_confidence, include_spans: include_spans)
      end
    end

    private

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

    def format_span(span, include_confidence:, include_spans:)
      return nil if span.nil?
      return span.text unless include_confidence || include_spans

      result = { 'text' => span.text }
      result['confidence'] = span.score if include_confidence
      if include_spans
        result['start'] = span.start
        result['end'] = span.end
      end
      result
    end
  end
end
