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
          else
            state[:descriptions][name] = config.to_s
          end
        end
      end
    end
  end
end
