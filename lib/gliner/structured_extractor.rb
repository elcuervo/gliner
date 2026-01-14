# frozen_string_literal: true

require_relative 'span'

module Gliner
  class StructuredExtractor
    def initialize(span_extractor)
      @span_extractor = span_extractor
    end

    def apply_choice_filters(spans_by_label, parsed_fields)
      filtered = spans_by_label.transform_values(&:dup)

      parsed_fields.each do |field|
        choices = field[:choices]
        next if choices.nil? || choices.empty?

        label = field[:name]
        spans = filtered.fetch(label, [])
        filtered[label] = filter_spans_by_choices(spans, choices)
      end

      filtered
    end

    def filter_spans_by_choices(spans, choices)
      return spans if spans.empty? || choices.nil? || choices.empty?

      normalized_choices = choices.map { |choice| normalize_choice(choice) }
      matched = spans.select { |span| normalized_choices.include?(normalize_choice(span.text)) }

      matched.empty? ? spans : matched
    end

    def build_structure_instances(parsed_fields, spans_by_label, include_confidence:, include_spans:)
      anchor_field = parsed_fields.find { |f| f[:dtype] == :str } || parsed_fields.first

      return [{}] if anchor_field.nil?

      anchors = spans_by_label.fetch(anchor_field[:name], [])

      if anchors.empty?
        return [format_structure_object(parsed_fields, spans_by_label,
                                        include_confidence: include_confidence,
                                        include_spans: include_spans)]
      end

      anchors_sorted = anchors.sort_by(&:start)
      instance_spans = anchors_sorted.map { Hash.new { |h, k| h[k] = [] } }

      spans_by_label.each do |label, spans|
        spans.each do |span|
          anchor_index = anchors_sorted.rindex { |anchor| anchor.start <= span.start } || 0
          instance_spans[anchor_index][label] << span
        end
      end

      instance_spans.map do |field_spans|
        format_structure_object(parsed_fields, field_spans,
                                include_confidence: include_confidence,
                                include_spans: include_spans)
      end
    end

    def format_structure_object(parsed_fields, spans_by_label, include_confidence:, include_spans:)
      obj = {}

      parsed_fields.each do |field|
        key = field[:name]
        spans = spans_by_label.fetch(key, [])

        if field[:dtype] == :str
          best = @span_extractor.choose_best_span(spans)
          obj[key] = @span_extractor.format_single_span(best, include_confidence: include_confidence, include_spans: include_spans)
        else
          obj[key] = @span_extractor.format_spans(spans, include_confidence: include_confidence, include_spans: include_spans)
        end
      end

      obj
    end

    private

    def normalize_choice(value)
      value.to_s.strip.downcase
    end
  end
end
