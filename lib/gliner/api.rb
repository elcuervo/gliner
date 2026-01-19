# frozen_string_literal: true

module Gliner
  module API
    class << self
      def compile(model, config)
        runner(config)
          .new(model, config)
      end

      private

      def runner(config)
        return StructuredRunner if structured_config?(config)

        EntityRunner
      end

      def structured_config?(config)
        return false unless config.is_a?(Hash)

        keys = config.transform_keys(&:to_s)
        return true if keys.key?('name') && keys.key?('fields')

        config.values.all? { |value| value.is_a?(Array) }
      end
    end

    class PreparedTask
      def initialize(task, parsed)
        @task = task
        @parsed = parsed
        @labels = task.labels(parsed)
        @schema_tokens = task.input_builder.schema_tokens_for(
          prompt: task.build_prompt(parsed),
          labels: @labels,
          label_prefix: task.label_prefix
        )
        @label_mask = Array.new(@labels.length, 1)
        @label_positions_template = precompute_label_positions
      end

      def call(text, **options)
        prepared = @task.input_builder.prepare(text, @schema_tokens)
        label_positions = @label_positions_template
        if label_positions.any? { |pos| pos.nil? || pos >= prepared.input_ids.length }
          label_positions = @task.inference.label_positions_for(prepared.word_ids, @labels.length)
        end
        logits = @task.inference.run(
          Inference::Request.new(
            input_ids: prepared.input_ids,
            attention_mask: prepared.attention_mask,
            words_mask: prepared.words_mask,
            text_lengths: [prepared.text_len],
            task_type: @task.task_type,
            label_positions: label_positions,
            label_mask: @label_mask,
            want_cls: @task.needs_cls_logits?
          )
        )

        @task.process_output(logits, @parsed, prepared, options.merge(label_positions: label_positions))
      end

      private

      def precompute_label_positions
        return [] if @labels.empty?

        prepared = @task.input_builder.prepare('.', @schema_tokens)
        @task.inference.label_positions_for(prepared.word_ids, @labels.length)
      end
    end

    class EntityRunner
      def initialize(model, config)
        parsed = model.entity_task.parse_config(config)
        @task = PreparedTask.new(model.entity_task, parsed)
      end

      def [](text, **options)
        result = @task.call(text, **options)
        result.fetch('entities')
      end

      alias call []
    end

    class StructuredRunner
      def initialize(model, config)
        @tasks = build_tasks(model, config)
      end

      def [](text, **options)
        @tasks.each_with_object({}) do |(name, task), out|
          out[name] = task.call(text, **options)
        end
      end

      alias call []

      private

      def build_tasks(model, config)
        raise Error, 'structures must be a Hash' unless config.is_a?(Hash)

        if config.key?(:name) || config.key?('name')
          parsed = model.json_task.parse_config(config)
          { parsed[:name].to_s => PreparedTask.new(model.json_task, parsed) }
        else
          config.each_with_object({}) do |(name, fields), tasks|
            parsed = model.json_task.parse_config(name: name, fields: fields)
            tasks[name.to_s] = PreparedTask.new(model.json_task, parsed)
          end
        end
      end
    end

    class ClassificationRunner
      def initialize(model, tasks_config)
        raise Error, 'tasks must be a Hash' unless tasks_config.is_a?(Hash)

        @tasks = tasks_config.map do |name, config|
          parsed = model.classification_task.parse_config(name: name, config: config)
          [name.to_s, PreparedTask.new(model.classification_task, parsed)]
        end
      end

      def [](text, **options)
        @tasks.each_with_object({}) do |(name, task), out|
          out[name] = task.call(text, **options)
        end
      end

      alias call []
    end

    class ClassificationProxy
      def initialize(model)
        @model = model
      end

      def [](tasks)
        ClassificationRunner.new(@model, tasks)
      end
    end
  end
end
