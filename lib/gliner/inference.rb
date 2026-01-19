# frozen_string_literal: true

module Gliner
  class Inference
    Request = Data.define(
      :input_ids,
      :attention_mask,
      :words_mask,
      :text_lengths,
      :task_type,
      :label_positions,
      :label_mask,
      :want_cls
    )

    TASK_TYPE_ENTITIES = 0
    TASK_TYPE_CLASSIFICATION = 1
    TASK_TYPE_JSON = 2

    SCHEMA_PREFIX_LENGTH = 4
    LABEL_SPACING = 2

    attr_reader :label_index_mode, :has_cls_logits

    def initialize(session)
      @session = session
      validate_io!
    end

    def run(request)
      text_lengths = Array(request.text_lengths).flatten

      inputs = {
        input_ids: [request.input_ids],
        attention_mask: [request.attention_mask],
        words_mask: [request.words_mask],
        text_lengths: text_lengths,
        task_type: [request.task_type],
        label_positions: [request.label_positions],
        label_mask: [request.label_mask]
      }

      if @input_names&.include?('token_type_ids')
        inputs[:token_type_ids] = [Array.new(request.input_ids.length, 0)]
      end

      inputs.select! { |name, _| @input_names.include?(name.to_s) } if @input_names

      output_names = [@output_name]
      output_names << 'cls_logits' if request.want_cls && @has_cls_logits
      out = @session.run(output_names, inputs)

      return { logits: out.fetch(0), cls_logits: out.fetch(1) } if output_names.length > 1

      out.fetch(0)
    end

    def label_positions_for(word_ids, label_count)
      label_count.times.map do |i|
        combined_idx = SCHEMA_PREFIX_LENGTH + (i * LABEL_SPACING)
        pos = word_ids.index(combined_idx)

        raise Error, "Could not locate label position at combined index #{combined_idx}" if pos.nil?

        pos
      end
    end

    def label_logit(logits, pos, width, label_index, label_positions)
      if @label_index_mode == :label_position
        raise Error, 'Label positions required for span_logits output' if label_positions.nil?

        label_pos = label_positions.fetch(label_index)
        logits[0][pos][width][label_pos]
      else
        logits[0][pos][width][label_index]
      end
    end

    def sigmoid(x)
      1.0 / (1.0 + Math.exp(-x))
    end

    def softmax(values)
      max_value = values.max
      exps = values.map { |value| Math.exp(value - max_value) }
      sum = exps.sum
      exps.map { |value| value / sum }
    end

    private

    def validate_io!
      @input_names = @session.inputs.map { |i| i[:name] }

      output_names = @session.outputs.map { |o| o[:name] }

      @has_cls_logits = output_names.include?('cls_logits')

      if output_names.include?('logits')
        @output_name = 'logits'
        @label_index_mode = :label_index

        expected_inputs = %w[input_ids attention_mask words_mask text_lengths task_type label_positions label_mask]
        missing = expected_inputs - @input_names

        raise Error, "Model missing inputs: #{missing.join(', ')}" unless missing.empty?
      elsif output_names.include?('span_logits')
        @output_name = 'span_logits'
        @label_index_mode = :label_position

        expected_inputs = %w[input_ids attention_mask]
        missing = expected_inputs - @input_names

        raise Error, "Model missing inputs: #{missing.join(', ')}" unless missing.empty?
      else
        raise Error, 'Model missing output: logits or span_logits'
      end
    end
  end
end
