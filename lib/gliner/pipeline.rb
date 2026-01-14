# frozen_string_literal: true

module Gliner
  # Pipeline executes tasks with a unified flow
  # Eliminates duplication across entity extraction, classification, and JSON extraction
  class Pipeline
    def initialize(text_processor:, inference:)
      @text_processor = text_processor
      @inference = inference
    end

    # Execute a task with the given input and options
    # @param task [Task] The task to execute
    # @param text [String] Input text
    # @param config User configuration (format depends on task)
    # @param options [Hash] Execution options (threshold, include_confidence, etc.)
    # @return Result specific to task type
    def execute(task, text, config, **options)
      # 1. Parse configuration
      parsed = task.parse_config(config)

      # 2. Prepare text
      prepared_text = task.normalize_text? ? @text_processor.normalize_text(text) : text.to_s

      # 3. Build prompt and schema tokens
      prompt = task.build_prompt(parsed)
      labels = task.labels(parsed)
      schema_tokens = task.input_builder.schema_tokens_for(
        prompt: prompt,
        labels: labels,
        label_prefix: task.label_prefix
      )

      # 4. Prepare inputs
      prepared = task.input_builder.prepare(
        prepared_text,
        schema_tokens,
        already_normalized: task.normalize_text?
      )

      # 5. Get label positions
      label_positions = @inference.label_positions_for(prepared[:word_ids], labels.length)

      # 6. Run inference
      logits = @inference.run(
        input_ids: prepared[:input_ids],
        attention_mask: prepared[:attention_mask],
        words_mask: prepared[:words_mask],
        text_lengths: [prepared[:text_len]],
        task_type: task.task_type,
        label_positions: label_positions,
        label_mask: Array.new(labels.length, 1),
        want_cls: task.needs_cls_logits?
      )

      # 7. Process output
      task.process_output(logits, parsed, prepared, options)
    end
  end
end
