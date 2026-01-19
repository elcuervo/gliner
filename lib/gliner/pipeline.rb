# frozen_string_literal: true

module Gliner
  class Pipeline
    def initialize(text_processor:, inference:)
      @text_processor = text_processor
      @inference = inference
    end

    def execute(task, text, config, **options)
      parsed = task.parse_config(config)
      prepared_text = prepare_text(task, text)
      labels = task.labels(parsed)
      prepared = prepare_input(task, prepared_text, parsed, labels)
      label_positions = label_positions_for(prepared, labels.length)
      logits = run_inference(task, prepared, labels, label_positions)

      task.process_output(logits, parsed, prepared, options.merge(label_positions: label_positions))
    end

    private

    def prepare_text(task, text)
      task.normalize_text? ? @text_processor.normalize_text(text) : text.to_s
    end

    def prepare_input(task, prepared_text, parsed, labels)
      schema_tokens = task.input_builder.schema_tokens_for(
        prompt: task.build_prompt(parsed),
        labels: labels,
        label_prefix: task.label_prefix
      )

      task.input_builder.prepare(
        prepared_text,
        schema_tokens,
        already_normalized: task.normalize_text?
      )
    end

    def label_positions_for(prepared, label_count)
      @inference.label_positions_for(prepared.word_ids, label_count)
    end

    def run_inference(task, prepared, labels, label_positions)
      @inference.run(build_request(task, prepared, labels, label_positions))
    end

    def build_request(task, prepared, labels, label_positions)
      Inference::Request.new(
        input_ids: prepared.input_ids,
        attention_mask: prepared.attention_mask,
        words_mask: prepared.words_mask,
        text_lengths: [prepared.text_len],
        task_type: task.task_type,
        label_positions: label_positions,
        label_mask: Array.new(labels.length, 1),
        want_cls: task.needs_cls_logits?
      )
    end
  end
end
