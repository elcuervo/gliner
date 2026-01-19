# frozen_string_literal: true

require 'gliner/config/entity_types'
require 'gliner/config/classification_task'
require 'gliner/config/field_spec'

module Gliner
  class ConfigParser
    def parse_entity_types(entity_types)
      Config::EntityTypes.parse(entity_types)
    end

    def parse_classification_task(task_name, config)
      Config::ClassificationTask.parse(task_name, config)
    end

    def parse_field_spec(spec)
      Config::FieldSpec.parse(spec)
    end

    def build_field_descriptions(parsed_fields)
      Config::FieldSpec.build_descriptions(parsed_fields)
    end

    def build_prompt(base, label_descriptions)
      prompt = base.to_s

      label_descriptions.to_h.each do |label, description|
        next if description.to_s.empty?

        prompt += " [DESCRIPTION] #{label}: #{description}"
      end

      prompt
    end
  end
end
