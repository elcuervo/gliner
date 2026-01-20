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
        threshold = options.fetch(:threshold, Gliner.config.threshold)
        label_positions = options[:label_positions] || inference.label_positions_for(prepared.word_ids, parsed[:labels].length)

        spans_by_label = extract_spans(logits, parsed, prepared, label_positions, threshold)

        { 'entities' => format_entities(parsed, spans_by_label) }
      end

      private

      def extract_spans(logits, parsed, prepared, label_positions, threshold)
        @span_extractor.extract_spans_by_label(
          logits,
          parsed[:labels],
          label_positions,
          prepared,
          threshold: threshold,
          thresholds_by_label: parsed[:thresholds]
        )
      end

      def format_entities(parsed, spans_by_label)
        parsed[:labels].each_with_object({}) do |label, entities|
          spans = spans_by_label.fetch(label)
          dtype = parsed[:dtypes].fetch(label, :list)

          entities[label] = format_entity_value(label, spans, dtype)
        end
      end

      def format_entity_value(label, spans, dtype)
        if dtype == :str
          @span_extractor.format_single_span(
            @span_extractor.choose_best_span(spans),
            label: label
          )
        else
          @span_extractor.format_spans(spans, label: label)
        end
      end
    end
  end
end
