# frozen_string_literal: true

module Gliner
  class SpanExtractor
    def initialize(inference, max_width:)
      @inference = inference
      @max_width = max_width
    end

    def extract_spans_by_label(logits:, labels:, label_positions:, pos_to_word_index:, start_map:, end_map:,
                                original_text:, text_len:, threshold:, thresholds_by_label: nil)
      out = {}
      labels.each_with_index do |label, label_index|
        label_threshold = threshold
        if thresholds_by_label && thresholds_by_label.key?(label.to_s)
          label_threshold = thresholds_by_label.fetch(label.to_s)
        end
        out[label.to_s] = find_spans_for_label(
          logits: logits,
          label_index: label_index,
          label_positions: label_positions,
          pos_to_word_index: pos_to_word_index,
          start_map: start_map,
          end_map: end_map,
          original_text: original_text,
          text_len: text_len,
          threshold: label_threshold
        )
      end
      out
    end

    def find_spans_for_label(logits:, label_index:, label_positions:, pos_to_word_index:, start_map:, end_map:,
                              original_text:, text_len:, threshold:)
      spans = []

      seq_len = logits[0].length
      (0...seq_len).each do |pos|
        start_word = pos_to_word_index[pos]
        next if start_word.nil?

        (0...@max_width).each do |width|
          end_word = start_word + width
          next if end_word >= text_len

          score = @inference.sigmoid(@inference.label_logit(logits, pos, width, label_index, label_positions))
          next if score < threshold

          char_start = start_map[start_word]
          char_end = end_map[end_word]
          next if char_start.nil? || char_end.nil?

          text_span = original_text[char_start...char_end].to_s.strip
          next if text_span.empty?

          spans << [text_span, score, char_start, char_end]
        end
      end

      spans
    end

    def choose_best_span(spans)
      return nil if spans.empty?
      sorted = spans.sort_by { |(t, score, start_pos, end_pos)| [-score, (end_pos - start_pos), t.length] }
      best = sorted[0]
      best_score = best[1]
      near = sorted.take_while { |s| (best_score - s[1]) <= 0.02 }
      near.min_by { |(t, score, start_pos, end_pos)| [(end_pos - start_pos), -score, t.length] } || best
    end

    def format_single_span(span, include_confidence:, include_spans:)
      return nil if span.nil?
      text, score, start_pos, end_pos = span

      if include_spans && include_confidence
        { "text" => text, "confidence" => score, "start" => start_pos, "end" => end_pos }
      elsif include_spans
        { "text" => text, "start" => start_pos, "end" => end_pos }
      elsif include_confidence
        { "text" => text, "confidence" => score }
      else
        text
      end
    end

    def format_spans(spans, include_confidence:, include_spans:)
      return [] if spans.empty?

      sorted = spans.sort_by { |(_, score, _, _)| -score }
      selected = []

      sorted.each do |text, score, start_pos, end_pos|
        overlaps = selected.any? { |(_, _, s, e)| !(end_pos <= s || start_pos >= e) }
        next if overlaps
        selected << [text, score, start_pos, end_pos]
      end

      if include_spans && include_confidence
        selected.map { |t, s, st, en| { "text" => t, "confidence" => s, "start" => st, "end" => en } }
      elsif include_spans
        selected.map { |t, _s, st, en| { "text" => t, "start" => st, "end" => en } }
      elsif include_confidence
        selected.map { |t, s, _st, _en| { "text" => t, "confidence" => s } }
      else
        selected.map(&:first)
      end
    end

    def pos_to_word_index_for(prepared, logits)
      seq_len = logits[0].length
      return (0...prepared[:text_len]).to_a if seq_len == prepared[:text_len]
      prepared[:pos_to_word_index]
    end
  end
end
