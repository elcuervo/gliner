# frozen_string_literal: true

module Gliner
  module Tasks
    class EntityExtraction < Task
      def initialize(config_parser:, inference:, input_builder:, span_extractor:)
        super(config_parser: config_parser, inference: inference, input_builder: input_builder)
        @span_extractor = span_extractor
      end

      def parse_config(input)
        config_parser.parse_entity_types(input)
      end

      def task_type
        Inference::TASK_TYPE_ENTITIES
      end

      def label_prefix
        '[E]'
      end

      def build_prompt(parsed)
        config_parser.build_prompt('entities', parsed[:descriptions])
      end

      def labels(parsed)
        parsed[:labels]
      end

      def process_output(logits, parsed, prepared, options)
        settings = entity_settings(options)
        label_positions = label_positions_for(prepared, parsed, settings)
        spans_by_label = extract_spans(logits, parsed, prepared, label_positions, settings)

        { 'entities' => format_entities(parsed, spans_by_label, settings) }
      end

      private

      def entity_settings(options)
        {
          threshold: options.fetch(:threshold, 0.5),
          include_confidence: options.fetch(:include_confidence, false),
          include_spans: options.fetch(:include_spans, false),
          label_positions: options[:label_positions]
        }
      end

      def label_positions_for(prepared, parsed, settings)
        settings[:label_positions] || inference.label_positions_for(prepared.word_ids, parsed[:labels].length)
      end

      def extract_spans(logits, parsed, prepared, label_positions, settings)
        @span_extractor.extract_spans_by_label(
          logits,
          parsed[:labels],
          label_positions,
          prepared,
          threshold: settings[:threshold],
          thresholds_by_label: parsed[:thresholds]
        )
      end

      def format_entities(parsed, spans_by_label, settings)
        parsed[:labels].each_with_object({}) do |label, entities|
          spans = spans_by_label.fetch(label)
          dtype = parsed[:dtypes].fetch(label, :list)

          entities[label] = format_entity_value(spans, dtype, settings)
        end
      end

      def format_entity_value(spans, dtype, settings)
        if dtype == :str
          @span_extractor.format_single_span(
            @span_extractor.choose_best_span(spans),
            include_confidence: settings[:include_confidence],
            include_spans: settings[:include_spans]
          )
        else
          @span_extractor.format_spans(
            spans,
            include_confidence: settings[:include_confidence],
            include_spans: settings[:include_spans]
          )
        end
      end
    end
  end
end
