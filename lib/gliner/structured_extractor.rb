# frozen_string_literal: true

module Gliner
  class StructuredExtractor
    def initialize(span_extractor)
      @span_extractor = span_extractor
    end

    def apply_choice_filters(spans_by_label, parsed_fields)
      filtered = spans_by_label.transform_values(&:dup)

      parsed_fields.each do |field|
        next unless field[:choices]&.any?

        label = field[:name]
        spans = filtered.fetch(label, [])
        filtered[label] = filter_spans_by_choices(spans, field[:choices])
      end

      filtered
    end

    def filter_spans_by_choices(spans, choices)
      return spans if spans.empty? || choices.nil? || choices.empty?

      normalized_choices = choices.map { |choice| normalize_choice(choice) }
      matched = spans.select { |span| normalized_choices.include?(normalize_choice(span.text)) }

      return spans if matched.empty?

      matched
    end

    def build_structure_instances(parsed_fields, spans_by_label, opts)
      format_opts = FormatOptions.from(opts)
      anchor_field = anchor_field_for(parsed_fields)
      return [Structure.new(fields: {})] unless anchor_field

      anchors = spans_by_label.fetch(anchor_field[:name], [])
      return [format_structure_object(parsed_fields, spans_by_label, format_opts)] if anchors.empty?

      instance_spans = build_instance_spans(anchors, spans_by_label)
      format_instances(parsed_fields, instance_spans, format_opts)
    end

    def format_structure_object(parsed_fields, spans_by_label, _opts)
      obj = {}

      parsed_fields.each do |field|
        key = field[:name]
        spans = spans_by_label.fetch(key, [])

        if field[:dtype] == :str
          best = @span_extractor.choose_best_span(spans)
          obj[key] = @span_extractor.format_single_span(best, label: key)
        else
          obj[key] = @span_extractor.format_spans(spans, label: key)
        end
      end

      Structure.new(fields: obj)
    end

    private

    def anchor_field_for(parsed_fields)
      parsed_fields.find { |field| field[:dtype] == :str } || parsed_fields.first
    end

    def build_instance_spans(anchors, spans_by_label)
      anchors_sorted = anchors.sort_by(&:start)
      instance_spans = anchors_sorted.map { Hash.new { |hash, key| hash[key] = [] } }

      spans_by_label.each do |label, spans|
        spans.each do |span|
          anchor_index = anchors_sorted.rindex { |anchor| anchor.start <= span.start } || 0
          instance_spans[anchor_index][label] << span
        end
      end

      instance_spans
    end

    def format_instances(parsed_fields, instance_spans, opts)
      instance_spans.map do |field_spans|
        format_structure_object(parsed_fields, field_spans, opts)
      end
    end

    def normalize_choice(value)
      value.to_s.strip.downcase
    end
  end
end
