# frozen_string_literal: true

module Gliner
  module Config
    class EntityTypes
      class << self
        def parse(entity_types)
          case entity_types
          when Array
            list_config(entity_types)
          when String, Symbol
            list_config([entity_types])
          when Hash
            hash_config(entity_types)
          else
            raise Error, 'labels must be a String, Array, or Hash'
          end
        end

        private

        def list_config(entity_types)
          {
            labels: entity_types.map(&:to_s),
            descriptions: {},
            dtypes: {},
            thresholds: {}
          }
        end

        def hash_config(entity_types)
          state = { labels: [], descriptions: {}, dtypes: {}, thresholds: {} }
          entity_types.each { |label, config| apply_config(state, label, config) }
          state
        end

        def apply_config(state, label, config)
          name = label.to_s
          state[:labels] << name

          return if config.nil?

          case config
          when String
            apply_description(state, name, config)
          when Hash
            apply_hash_config(state, name, config)
          else
            apply_description(state, name, config.to_s)
          end
        end

        def apply_hash_config(state, name, config)
          config_hash = config.transform_keys(&:to_s)
          apply_description(state, name, config_hash['description']) if config_hash['description']
          apply_dtype(state, name, config_hash['dtype']) if config_hash['dtype']
          apply_threshold(state, name, config_hash['threshold']) if config_hash.key?('threshold')
        end

        def apply_description(state, name, description)
          state[:descriptions][name] = description
        end

        def apply_dtype(state, name, dtype)
          state[:dtypes][name] = dtype.to_s == 'str' ? :str : :list
        end

        def apply_threshold(state, name, threshold)
          state[:thresholds][name] = Float(threshold)
        end
      end
    end
  end
end
