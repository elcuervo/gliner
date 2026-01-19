# frozen_string_literal: true

module Gliner
  module Tasks
    class Classification < Task
      def initialize(config_parser:, inference:, input_builder:, classifier:)
        super(config_parser: config_parser, inference: inference, input_builder: input_builder)

        @classifier = classifier
      end

      def parse_config(input)
        raise Error, 'classification config must be a Hash' unless input.is_a?(Hash)

        name, task_config = extract_task_config(input)
        parsed = config_parser.parse_classification_task(name, task_config)
        parsed.merge(name: name.to_s)
      end

      def task_type
        Inference::TASK_TYPE_CLASSIFICATION
      end

      def label_prefix
        '[L]'
      end

      def build_prompt(parsed)
        config_parser.build_prompt(parsed[:name], parsed[:label_descs])
      end

      def labels(parsed)
        parsed[:labels]
      end

      def needs_cls_logits?
        inference.has_cls_logits
      end

      def process_output(logits, parsed, prepared, options)
        include_confidence = options.fetch(:include_confidence, false)
        threshold_override = options[:threshold]
        cls_threshold = threshold_override.nil? ? parsed[:cls_threshold] : threshold_override

        scores = classification_scores(logits, parsed, prepared, options)
        @classifier.format_classification(
          scores,
          labels: parsed[:labels],
          multi_label: parsed[:multi_label],
          include_confidence: include_confidence,
          cls_threshold: cls_threshold
        )
      end

      def execute_all(pipeline, text, tasks_config, **options)
        raise Error, 'tasks must be a Hash' unless tasks_config.is_a?(Hash)

        tasks_config.each_with_object({}) do |(task_name, task_config), results|
          parsed_config = { name: task_name, config: task_config }
          results[task_name.to_s] = pipeline.execute(self, text, parsed_config, **options)
        end
      end

      private

      def extract_task_config(input)
        name = input[:name] || input['name']
        task_config = input[:config] || input['config']

        return [name, task_config] if name && task_config
        return input.first if name.nil? && task_config.nil? && input.length == 1

        raise Error, 'classification config must include :name and :config'
      end

      def classification_scores(logits, parsed, prepared, options)
        return cls_scores(logits, parsed) if cls_logits?(logits)

        label_positions = options.fetch(:label_positions) do
          inference.label_positions_for(prepared.word_ids, parsed[:labels].length)
        end

        @classifier.classification_scores(
          logits,
          parsed[:labels],
          label_positions,
          prepared
        )
      end

      def cls_logits?(logits)
        logits.is_a?(Hash) && logits.key?(:cls_logits)
      end

      def cls_scores(logits, parsed)
        cls_logits = Array(logits.fetch(:cls_logits).fetch(0))
        parsed[:multi_label] ? cls_logits.map { |value| inference.sigmoid(value) } : inference.softmax(cls_logits)
      end
    end
  end
end
