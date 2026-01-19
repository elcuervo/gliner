# frozen_string_literal: true

require 'gliner/inference/request'
require 'gliner/inference/io_validator'

module Gliner
  class Inference
    TASK_TYPE_ENTITIES = 0
    TASK_TYPE_CLASSIFICATION = 1
    TASK_TYPE_JSON = 2

    SCHEMA_PREFIX_LENGTH = 4
    LABEL_SPACING = 2

    attr_reader :label_index_mode, :has_cls_logits

    def initialize(session)
      @session = session
      validation = IOValidator.call(session)
      @input_names = validation.input_names
      @output_name = validation.output_name
      @label_index_mode = validation.label_index_mode
      @has_cls_logits = validation.has_cls_logits
    end

    def run(request)
      outputs = output_names_for(request)
      out = @session.run(outputs, build_inputs(request))
      format_outputs(out, outputs)
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

    def sigmoid(value)
      1.0 / (1.0 + Math.exp(-value))
    end

    def softmax(values)
      max_value = values.max
      exps = values.map { |value| Math.exp(value - max_value) }
      sum = exps.sum
      exps.map { |value| value / sum }
    end

    private

    def build_inputs(request)
      inputs = base_inputs(request)
      add_token_type_ids(inputs, request)
      filter_inputs(inputs)
    end

    def base_inputs(request)
      {
        input_ids: [request.input_ids],
        attention_mask: [request.attention_mask],
        words_mask: [request.words_mask],
        text_lengths: Array(request.text_lengths).flatten,
        task_type: [request.task_type],
        label_positions: [request.label_positions],
        label_mask: [request.label_mask]
      }
    end

    def add_token_type_ids(inputs, request)
      return inputs unless @input_names&.include?('token_type_ids')

      inputs[:token_type_ids] = [Array.new(request.input_ids.length, 0)]
      inputs
    end

    def filter_inputs(inputs)
      return inputs unless @input_names

      inputs.select { |name, _| @input_names.include?(name.to_s) }
    end

    def output_names_for(request)
      output_names = [@output_name]
      output_names << 'cls_logits' if request.want_cls && @has_cls_logits
      output_names
    end

    def format_outputs(out, output_names)
      return { logits: out.fetch(0), cls_logits: out.fetch(1) } if output_names.length > 1

      out.fetch(0)
    end
  end
end
