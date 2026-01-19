# frozen_string_literal: true

module Gliner
  class Inference
    IOValidation = Data.define(:input_names, :output_name, :label_index_mode, :has_cls_logits)

    class IOValidator
      class << self
        def call(session)
          input_names = session.inputs.map { |input| input[:name] }
          output_names = session.outputs.map { |output| output[:name] }
          has_cls_logits = output_names.include?('cls_logits')

          validation = validation_for_outputs(output_names, input_names)

          IOValidation.new(
            input_names: input_names,
            output_name: validation.fetch(:output_name),
            label_index_mode: validation.fetch(:label_index_mode),
            has_cls_logits: has_cls_logits
          )
        end

        private

        def validation_for_outputs(output_names, input_names)
          return validation_for_logits(input_names) if output_names.include?('logits')
          return validation_for_span_logits(input_names) if output_names.include?('span_logits')

          raise Error, 'Model missing output: logits or span_logits'
        end

        def validation_for_logits(input_names)
          expected_inputs = %w[input_ids attention_mask words_mask text_lengths task_type label_positions label_mask]
          ensure_expected_inputs!(expected_inputs, input_names)

          { output_name: 'logits', label_index_mode: :label_index }
        end

        def validation_for_span_logits(input_names)
          expected_inputs = %w[input_ids attention_mask]
          ensure_expected_inputs!(expected_inputs, input_names)

          { output_name: 'span_logits', label_index_mode: :label_position }
        end

        def ensure_expected_inputs!(expected_inputs, input_names)
          missing = expected_inputs - input_names
          raise Error, "Model missing inputs: #{missing.join(', ')}" unless missing.empty?
        end
      end
    end
  end
end
