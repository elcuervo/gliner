# frozen_string_literal: true

module Gliner
  class ConfigParser
    def parse_entity_types(entity_types)
      case entity_types
      when Array
        entity_list_config(entity_types)
      when String, Symbol
        entity_list_config([entity_types])
      when Hash
        entity_hash_config(entity_types)
      else
        raise Error, 'labels must be a String, Array, or Hash'
      end
    end

    def parse_classification_task(task_name, config)
      case config
      when Array
        classification_from_labels(config)
      when Hash
        classification_from_hash(task_name, config)
      else
        raise Error, "classification task #{task_name.inspect} must be an Array or Hash"
      end
    end

    def parse_field_spec(spec)
      name, *parts = spec.split('::')
      field = {
        name: name.to_s,
        dtype: :list,
        description: nil,
        choices: nil,
        dtype_explicit: false
      }

      parts.each { |part| apply_field_part(field, part) }
      field.delete(:dtype_explicit)
      field
    end

    def build_field_descriptions(parsed_fields)
      parsed_fields.each_with_object({}) do |field, acc|
        description = field_description(field)
        acc[field[:name]] = description if description
      end
    end

    def build_prompt(base, label_descriptions)
      prompt = base.to_s
      label_descriptions.to_h.each do |label, description|
        next if description.to_s.empty?

        prompt += " [DESCRIPTION] #{label}: #{description}"
      end
      prompt
    end

    private

    def entity_list_config(entity_types)
      {
        labels: entity_types.map(&:to_s),
        descriptions: {},
        dtypes: {},
        thresholds: {}
      }
    end

    def entity_hash_config(entity_types)
      state = { labels: [], descriptions: {}, dtypes: {}, thresholds: {} }
      entity_types.each { |label, config| apply_entity_config(state, label, config) }
      state
    end

    def apply_entity_config(state, label, config)
      name = label.to_s
      state[:labels] << name

      case config
      when String
        state[:descriptions][name] = config
      when Hash
        config_hash = config.transform_keys(&:to_s)
        description = config_hash['description']
        state[:descriptions][name] = description.to_s if description
        dtype = config_hash['dtype']
        state[:dtypes][name] = dtype.to_s == 'str' ? :str : :list if dtype
        state[:thresholds][name] = Float(config_hash['threshold']) if config_hash.key?('threshold')
      when nil
        # ignore
      else
        state[:descriptions][name] = config.to_s
      end
    end

    def classification_from_labels(labels)
      {
        labels: labels.map(&:to_s),
        multi_label: false,
        cls_threshold: 0.5,
        label_descs: {}
      }
    end

    def classification_from_hash(task_name, config)
      config_hash = config.transform_keys(&:to_s)

      return classification_from_described_labels(task_name, config_hash) if config_hash.key?('labels')

      {
        labels: config.keys.map(&:to_s),
        multi_label: false,
        cls_threshold: 0.5,
        label_descs: config.transform_keys(&:to_s).transform_values(&:to_s)
      }
    end

    def classification_from_described_labels(task_name, config_hash)
      labels, label_descs = parse_classification_labels(task_name, config_hash['labels'])

      {
        labels: labels,
        multi_label: !!config_hash['multi_label'],
        cls_threshold: classification_threshold(config_hash['cls_threshold']),
        label_descs: label_descs
      }
    end

    def parse_classification_labels(task_name, raw_labels)
      case raw_labels
      when Array
        [raw_labels.map(&:to_s), {}]
      when Hash
        [raw_labels.keys.map(&:to_s), raw_labels.transform_keys(&:to_s).transform_values(&:to_s)]
      else
        raise Error, "classification task #{task_name.inspect} must include labels"
      end
    end

    def classification_threshold(value)
      return 0.5 if value.nil? || value == false

      Float(value)
    end

    def apply_field_part(field, part)
      part = part.to_s
      case part
      when 'str'
        set_field_dtype(field, :str)
      when 'list'
        set_field_dtype(field, :list)
      else
        if bracketed_list?(part)
          field[:choices] = parse_choices(part)
          field[:dtype] = :str unless field[:dtype_explicit]
        else
          append_field_description(field, part)
        end
      end
    end

    def set_field_dtype(field, dtype)
      field[:dtype] = dtype
      field[:dtype_explicit] = true
    end

    def append_field_description(field, part)
      field[:description] = [field[:description], part].compact.join('::')
    end

    def bracketed_list?(part)
      part.start_with?('[') && part.end_with?(']')
    end

    def parse_choices(part)
      part[1..-2].split('|').map(&:strip).reject(&:empty?)
    end

    def field_description(field)
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
