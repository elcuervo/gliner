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
        threshold = options.fetch(:threshold, 0.5)
        include_confidence = options.fetch(:include_confidence, false)
        include_spans = options.fetch(:include_spans, false)

        label_positions = options.fetch(:label_positions) do
          inference.label_positions_for(prepared.word_ids, parsed[:labels].length)
        end

        spans_by_label = @span_extractor.extract_spans_by_label(
          logits,
          parsed[:labels],
          label_positions,
          prepared,
          threshold: threshold,
          thresholds_by_label: parsed[:thresholds]
        )

        entities = {}
        parsed[:labels].each do |label|
          spans = spans_by_label.fetch(label)
          dtype = parsed[:dtypes].fetch(label, :list)

          entities[label] =
            if dtype == :str
              @span_extractor.format_single_span(
                @span_extractor.choose_best_span(spans),
                include_confidence: include_confidence,
                include_spans: include_spans
              )
            else
              @span_extractor.format_spans(spans, include_confidence: include_confidence, include_spans: include_spans)
            end
        end

        { 'entities' => entities }
      end
    end
  end
end
