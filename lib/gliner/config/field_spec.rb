# frozen_string_literal: true

module Gliner
  module Config
    class FieldSpec
      class << self
        def parse(spec)
          name, *parts = spec.split('::')
          field = build_field(name)

          parts.each { |part| apply_part(field, part.to_s) }
          field.delete(:dtype_explicit)
          field
        end

        def build_descriptions(parsed_fields)
          parsed_fields.each_with_object({}) do |field, acc|
            description = description_for(field)
            acc[field[:name]] = description if description
          end
        end

        private

        def build_field(name)
          {
            name: name.to_s,
            dtype: :list,
            description: nil,
            choices: nil,
            dtype_explicit: false
          }
        end

        def apply_part(field, part)
          return apply_dtype_part(field, part) if dtype_part?(part)
          return apply_choice_part(field, part) if bracketed_list?(part)

          append_description(field, part)
        end

        def dtype_part?(part)
          %w[str list].include?(part)
        end

        def apply_dtype_part(field, part)
          set_dtype(field, part == 'str' ? :str : :list)
        end

        def apply_choice_part(field, part)
          field[:choices] = parse_choices(part)
          field[:dtype] = :str unless field[:dtype_explicit]
        end

        def set_dtype(field, dtype)
          field[:dtype] = dtype
          field[:dtype_explicit] = true
        end

        def append_description(field, part)
          field[:description] = [field[:description], part].compact.join('::')
        end

        def bracketed_list?(part)
          part.start_with?('[') && part.end_with?(']')
        end

        def parse_choices(part)
          part[1..-2].split('|').map(&:strip).reject(&:empty?)
        end

        def description_for(field)
          description = field[:description].to_s
          choices = field[:choices]
          return nil if description.empty? && !choices&.any?

          if choices&.any?
            choices_str = choices.join('|')
            description = description.empty? ? "Choices: #{choices_str}" : "#{description} (choices: #{choices_str})"
          end

          description.empty? ? nil : description
        end
      end
    end
  end
end
