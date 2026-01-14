# frozen_string_literal: true

module Gliner
  module Tasks
    class Classification < Task
      def initialize(config_parser:, inference:, input_builder:, classifier:, span_extractor:)
        super(config_parser: config_parser, inference: inference, input_builder: input_builder)
        @classifier = classifier
        @span_extractor = span_extractor
        @current_task_name = nil
      end

      def parse_config(input)
        raise Error, "tasks must be a Hash" unless input.is_a?(Hash)

        # Store all task configs, return first for initial processing
        @task_configs = input.transform_keys(&:to_s).map do |task_name, config|
          [task_name, config_parser.parse_classification_task(task_name, config)]
        end

        @task_configs.first&.last
      end

      def task_type
        Inference::TASK_TYPE_CLASSIFICATION
      end

      def label_prefix
        "[L]"
      end

      def build_prompt(parsed)
        config_parser.build_prompt(@current_task_name, parsed[:label_descs])
      end

      def labels(parsed)
        parsed[:labels]
      end

      def needs_cls_logits?
        inference.has_cls_logits
      end

      def process_output(logits, parsed, prepared, options)
        # This is called for single task execution
        # For multi-task, we need different handling
        raise Error, "Classification task requires multi-task processing"
      end

      # Custom execution for classification that handles multiple tasks
      def execute_all(pipeline, text, tasks_config, **options)
        raise Error, "tasks must be a Hash" unless tasks_config.is_a?(Hash)

        results = {}
        tasks_config.each do |task_name, config|
          @current_task_name = task_name.to_s
          parsed = config_parser.parse_classification_task(task_name, config)

          prompt = config_parser.build_prompt(task_name.to_s, parsed[:label_descs])
          schema_tokens = input_builder.schema_tokens_for(
            prompt: prompt,
            labels: parsed[:labels],
            label_prefix: "[L]"
          )

          prepared = input_builder.prepare(text, schema_tokens)
          label_positions = inference.label_positions_for(prepared[:word_ids], parsed[:labels].length)

          scores =
            if inference.has_cls_logits
              out_logits = inference.run(
                input_ids: prepared[:input_ids],
                attention_mask: prepared[:attention_mask],
                words_mask: prepared[:words_mask],
                text_lengths: [prepared[:text_len]],
                task_type: Inference::TASK_TYPE_CLASSIFICATION,
                label_positions: label_positions,
                label_mask: Array.new(parsed[:labels].length, 1),
                want_cls: true
              )
              cls_logits = Array(out_logits.fetch(:cls_logits).fetch(0))
              parsed[:multi_label] ? cls_logits.map { |x| inference.sigmoid(x) } : inference.softmax(cls_logits)
            else
              logits = inference.run(
                input_ids: prepared[:input_ids],
                attention_mask: prepared[:attention_mask],
                words_mask: prepared[:words_mask],
                text_lengths: [prepared[:text_len]],
                task_type: Inference::TASK_TYPE_CLASSIFICATION,
                label_positions: label_positions,
                label_mask: Array.new(parsed[:labels].length, 1)
              )
              pos_to_word_index = @span_extractor.pos_to_word_index_for(prepared, logits)
              @classifier.classification_scores(
                logits: logits,
                labels: parsed[:labels],
                label_positions: label_positions,
                pos_to_word_index: pos_to_word_index,
                text_len: prepared[:text_len],
                threshold: parsed[:cls_threshold]
              )
            end

          results[task_name.to_s] = @classifier.format_classification(
            scores,
            labels: parsed[:labels],
            multi_label: parsed[:multi_label],
            include_confidence: options.fetch(:include_confidence, false),
            cls_threshold: parsed[:cls_threshold]
          )
        end

        results
      end
    end
  end
end
