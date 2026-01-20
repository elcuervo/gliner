# frozen_string_literal: true

module Gliner
  module Runners
    class StructuredRunner
      include Inspectable

      def initialize(model, config)
        @tasks = build_tasks(model, config)
      end

      def [](text, **options)
        @tasks.transform_values do |task|
          task.call(text, **options)
        end
      end

      alias call []

      private

      def inspect_label = 'Structure'
      def inspect_items = @tasks.keys

      def build_tasks(model, config)
        raise Error, 'structures must be a Hash' unless config.is_a?(Hash)

        if config.key?(:name) || config.key?('name')
          parsed = model.json_task.parse_config(config)

          { parsed[:name].to_s => PreparedTask.new(model.json_task, parsed) }
        else
          config.each_with_object({}) do |(name, fields), tasks|
            parsed = model.json_task.parse_config(name: name, fields: fields)
            tasks[name.to_s] = PreparedTask.new(model.json_task, parsed)
          end
        end
      end
    end
  end
end
