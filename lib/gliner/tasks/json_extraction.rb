# frozen_string_literal: true

module Gliner
  module Tasks
    class JsonExtraction < Task
      def initialize(config_parser:, inference:, input_builder:, span_extractor:, structured_extractor:, text_processor:)
        super(config_parser: config_parser, inference: inference, input_builder: input_builder)
        @span_extractor = span_extractor
        @structured_extractor = structured_extractor
        @text_processor = text_processor
      end

      def parse_config(input)
        raise Error, "structures must be a Hash" unless input.is_a?(Hash)

        # Store all structure configs
        @structure_configs = input.transform_keys(&:to_s).map do |parent, fields|
          parsed_fields = Array(fields).map { |spec| config_parser.parse_field_spec(spec.to_s) }
          [parent, parsed_fields]
        end

        @structure_configs.first&.last
      end

      def task_type
        Inference::TASK_TYPE_JSON
      end

      def label_prefix
        "[C]"
      end

      def normalize_text?
        true
      end

      def build_prompt(parsed)
        # Not used in standard flow - we handle this in execute_all
        ""
      end

      def labels(parsed)
        # Not used in standard flow - we handle this in execute_all
        []
      end

      def process_output(logits, parsed, prepared, options)
        # Not used - we use execute_all instead
        raise Error, "JsonExtraction task requires execute_all"
      end

      # Custom execution for JSON extraction that handles multiple structures
      def execute_all(pipeline, text, structures_config, **options)
        raise Error, "structures must be a Hash" unless structures_config.is_a?(Hash)

        normalized_text = @text_processor.normalize_text(text)
        results = {}

        structures_config.each do |parent, fields|
          parent_name = parent.to_s
          parsed_fields = Array(fields).map { |spec| config_parser.parse_field_spec(spec.to_s) }
          labels = parsed_fields.map { |f| f[:name] }
          descs = config_parser.build_field_descriptions(parsed_fields)

          prompt = config_parser.build_prompt(parent_name, descs)
          schema_tokens = input_builder.schema_tokens_for(
            prompt: prompt,
            labels: labels,
            label_prefix: "[C]"
          )

          prepared = input_builder.prepare(normalized_text, schema_tokens, already_normalized: true)
          label_positions = inference.label_positions_for(prepared[:word_ids], labels.length)

          logits = inference.run(
            input_ids: prepared[:input_ids],
            attention_mask: prepared[:attention_mask],
            words_mask: prepared[:words_mask],
            text_lengths: [prepared[:text_len]],
            task_type: Inference::TASK_TYPE_JSON,
            label_positions: label_positions,
            label_mask: Array.new(labels.length, 1)
          )

          pos_to_word_index = @span_extractor.pos_to_word_index_for(prepared, logits)

          spans_by_label = @span_extractor.extract_spans_by_label(
            logits: logits,
            labels: labels,
            label_positions: label_positions,
            pos_to_word_index: pos_to_word_index,
            start_map: prepared[:start_map],
            end_map: prepared[:end_map],
            original_text: prepared[:original_text],
            text_len: prepared[:text_len],
            threshold: options.fetch(:threshold, 0.5)
          )

          filtered_spans = @structured_extractor.apply_choice_filters(spans_by_label, parsed_fields)
          instances = @structured_extractor.build_structure_instances(
            parsed_fields,
            filtered_spans,
            include_confidence: options.fetch(:include_confidence, false),
            include_spans: options.fetch(:include_spans, false)
          )

          results[parent_name] = instances
        end

        results
      end
    end
  end
end
