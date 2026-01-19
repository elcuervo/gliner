# frozen_string_literal: true

module Gliner
  module Runners
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
  end
end
