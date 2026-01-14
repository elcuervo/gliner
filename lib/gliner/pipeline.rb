# frozen_string_literal: true

module Gliner
  class Pipeline
    def initialize(text_processor:, inference:)
      @text_processor = text_processor
      @inference = inference
    end

    def execute(task, text, config, **options)
      parsed = task.parse_config(config)
      prepared_text = task.normalize_text? ? @text_processor.normalize_text(text) : text.to_s
      prompt = task.build_prompt(parsed)
      labels = task.labels(parsed)

      schema_tokens = task.input_builder.schema_tokens_for(
        prompt: prompt,
        labels: labels,
        label_prefix: task.label_prefix
      )

      prepared = task.input_builder.prepare(
        prepared_text,
        schema_tokens,
        already_normalized: task.normalize_text?
      )

      label_positions = @inference.label_positions_for(prepared.word_ids, labels.length)

      logits = @inference.run(
        input_ids: prepared.input_ids,
        attention_mask: prepared.attention_mask,
        words_mask: prepared.words_mask,
        text_lengths: [prepared.text_len],
        task_type: task.task_type,
        label_positions: label_positions,
        label_mask: Array.new(labels.length, 1),
        want_cls: task.needs_cls_logits?
      )

      task.process_output(logits, parsed, prepared, options.merge(label_positions: label_positions))
    end
  end
end
