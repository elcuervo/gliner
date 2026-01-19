# frozen_string_literal: true

module Gliner
  module Config
    class FieldSpec
      class << self
        def parse(spec)
          name, *parts = spec.split('::')
          field = {
            name: name.to_s,
            dtype: :list,
            description: nil,
            choices: nil,
            dtype_explicit: false
          }

          parts.each { |part| apply_part(field, part) }
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

        def apply_part(field, part)
          part = part.to_s

          case part
          when 'str'
            set_dtype(field, :str)
          when 'list'
            set_dtype(field, :list)
          else
            if bracketed_list?(part)
              field[:choices] = parse_choices(part)
              field[:dtype] = :str unless field[:dtype_explicit]
            else
              append_description(field, part)
            end
          end
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
