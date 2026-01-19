# frozen_string_literal: true

require 'gliner/options'

module Gliner
  module Tasks
    class JsonExtraction < Task
      def initialize(config_parser:, inference:, input_builder:, span_extractor:, structured_extractor:)
        super(config_parser: config_parser, inference: inference, input_builder: input_builder)

        @span_extractor = span_extractor
        @structured_extractor = structured_extractor
      end

      def parse_config(input)
        raise Error, 'structure config must be a Hash' unless input.is_a?(Hash)

        name, fields = extract_structure_config(input)
        parsed_fields = Array(fields).map { |spec| config_parser.parse_field_spec(spec.to_s) }

        {
          name: name.to_s,
          parsed_fields: parsed_fields,
          labels: parsed_fields.map { |field| field[:name] },
          descriptions: config_parser.build_field_descriptions(parsed_fields)
        }
      end

      def task_type
        Inference::TASK_TYPE_JSON
      end

      def label_prefix
        '[C]'
      end

      def normalize_text?
        true
      end

      def build_prompt(parsed)
        config_parser.build_prompt(parsed[:name], parsed[:descriptions])
      end

      def labels(parsed)
        parsed[:labels]
      end

      def process_output(logits, parsed, prepared, options)
        spans_by_label = extract_spans(logits, parsed, prepared, options)
        filtered_spans = @structured_extractor.apply_choice_filters(spans_by_label, parsed[:parsed_fields])
        format_opts = FormatOptions.from(options)

        @structured_extractor.build_structure_instances(parsed[:parsed_fields], filtered_spans, format_opts)
      end

      def execute_all(pipeline, text, structures_config, **options)
        raise Error, 'structures must be a Hash' unless structures_config.is_a?(Hash)

        structures_config.each_with_object({}) do |(parent, fields), results|
          parsed_config = { name: parent, fields: fields }
          results[parent.to_s] = pipeline.execute(self, text, parsed_config, **options)
        end
      end

      private

      def extract_structure_config(input)
        name = input[:name] || input['name']
        fields = input[:fields] || input['fields']

        return [name, fields] if name && fields
        return input.first if name.nil? && fields.nil? && input.length == 1

        raise Error, 'structure config must include :name and :fields'
      end

      def extract_spans(logits, parsed, prepared, options)
        label_positions = options.fetch(:label_positions) do
          inference.label_positions_for(prepared.word_ids, parsed[:labels].length)
        end

        @span_extractor.extract_spans_by_label(
          logits,
          parsed[:labels],
          label_positions,
          prepared,
          threshold: options.fetch(:threshold, 0.5)
        )
      end
    end
  end
end
