# frozen_string_literal: true

module Gliner
  module Config
    class ClassificationTask
      DEFAULT_THRESHOLD = 0.5

      class << self
        def parse(task_name, config)
          case config
          when Array
            from_labels(config)
          when Hash
            from_hash(task_name, config)
          else
            raise Error, "classification task #{task_name.inspect} must be an Array or Hash"
          end
        end

        private

        def from_labels(labels)
          {
            labels: labels.map(&:to_s),
            multi_label: false,
            cls_threshold: DEFAULT_THRESHOLD,
            label_descs: {}
          }
        end

        def from_hash(task_name, config)
          config_hash = config.transform_keys(&:to_s)

          return from_described_labels(task_name, config_hash) if config_hash.key?('labels')

          {
            labels: config.keys.map(&:to_s),
            multi_label: false,
            cls_threshold: DEFAULT_THRESHOLD,
            label_descs: config.transform_keys(&:to_s).transform_values(&:to_s)
          }
        end

        def from_described_labels(task_name, config_hash)
          labels, label_descs = parse_labels(task_name, config_hash['labels'])

          {
            labels: labels,
            multi_label: config_hash['multi_label'] ? true : false,
            cls_threshold: threshold(config_hash['cls_threshold']),
            label_descs: label_descs
          }
        end

        def parse_labels(task_name, raw_labels)
          case raw_labels
          when Array
            [raw_labels.map(&:to_s), {}]
          when Hash
            [raw_labels.keys.map(&:to_s), raw_labels.transform_keys(&:to_s).transform_values(&:to_s)]
          else
            raise Error, "classification task #{task_name.inspect} must include labels"
          end
        end

        def threshold(value)
          return DEFAULT_THRESHOLD if value.nil? || value == false

          Float(value)
        end
      end
    end
  end
end
