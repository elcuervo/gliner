# frozen_string_literal: true

module Gliner
  class ConfigParser
    def parse_entity_types(entity_types)
      case entity_types
      when Array
        {
          labels: entity_types.map(&:to_s),
          descriptions: {},
          dtypes: {},
          thresholds: {}
        }
      when String, Symbol
        {
          labels: [entity_types.to_s],
          descriptions: {},
          dtypes: {},
          thresholds: {}
        }
      when Hash
        labels = []
        descriptions = {}
        dtypes = {}
        thresholds = {}

        entity_types.each do |label, config|
          name = label.to_s
          labels << name

          case config
          when String
            descriptions[name] = config
          when Hash
            cfg = config.transform_keys(&:to_s)
            if cfg["description"]
              descriptions[name] = cfg["description"].to_s
            end
            if cfg["dtype"]
              dtype = cfg["dtype"].to_s
              dtypes[name] = dtype == "str" ? :str : :list
            end
            if cfg.key?("threshold")
              thresholds[name] = Float(cfg["threshold"])
            end
          when nil
            # ignore
          else
            descriptions[name] = config.to_s
          end
        end

        {
          labels: labels,
          descriptions: descriptions,
          dtypes: dtypes,
          thresholds: thresholds
        }
      else
        raise Error, "labels must be a String, Array, or Hash"
      end
    end

    def parse_classification_task(task_name, config)
      multi_label = false
      cls_threshold = 0.5
      labels = []
      label_descs = {}

      case config
      when Array
        labels = config.map(&:to_s)
      when Hash
        cfg = config.transform_keys(&:to_s)
        if cfg.key?("labels")
          multi_label = !!cfg["multi_label"]
          cls_threshold = cfg["cls_threshold"] ? Float(cfg["cls_threshold"]) : cls_threshold

          raw_labels = cfg["labels"]
          if raw_labels.is_a?(Array)
            labels = raw_labels.map(&:to_s)
          elsif raw_labels.is_a?(Hash)
            labels = raw_labels.keys.map(&:to_s)
            label_descs = raw_labels.transform_keys(&:to_s).transform_values(&:to_s)
          else
            raise Error, "classification task #{task_name.inspect} must include labels"
          end
        else
          labels = config.keys.map(&:to_s)
          label_descs = config.transform_keys(&:to_s).transform_values(&:to_s)
        end
      else
        raise Error, "classification task #{task_name.inspect} must be an Array or Hash"
      end

      {
        labels: labels,
        multi_label: multi_label,
        cls_threshold: cls_threshold,
        label_descs: label_descs
      }
    end

    def parse_field_spec(spec)
      parts = spec.split("::")
      name = parts[0].to_s
      dtype = :list
      description = nil
      dtype_explicit = false
      choices = nil

      parts.drop(1).each do |part|
        part = part.to_s
        if part == "str"
          dtype = :str
          dtype_explicit = true
        elsif part == "list"
          dtype = :list
          dtype_explicit = true
        elsif part.start_with?("[") && part.end_with?("]")
          choices = part[1..-2].split("|").map(&:strip).reject(&:empty?)
          dtype = :str unless dtype_explicit
        elsif description.nil?
          description = part
        else
          description += "::#{part}"
        end
      end

      { name: name, dtype: dtype, description: description, choices: choices }
    end

    def build_field_descriptions(parsed_fields)
      parsed_fields.each_with_object({}) do |field, acc|
        desc = field[:description].to_s
        if field[:choices] && !field[:choices].empty?
          choices_str = field[:choices].join("|")
          desc = desc.empty? ? "Choices: #{choices_str}" : "#{desc} (choices: #{choices_str})"
        end
        acc[field[:name]] = desc unless desc.empty?
      end
    end

    def build_prompt(base, label_descriptions)
      prompt = base.to_s
      label_descriptions.to_h.each do |label, desc|
        next if desc.to_s.empty?
        prompt += " [DESCRIPTION] #{label}: #{desc}"
      end
      prompt
    end
  end
end
