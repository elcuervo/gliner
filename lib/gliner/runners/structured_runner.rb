# frozen_string_literal: true

module Gliner
  module Runners
    class StructuredRunner
      include Inspectable

      Result = Data.define(:name, :items) do
        def [](index) = items[index]
          def fetch(*args, &block) = items.fetch(*args, &block)
          def each(&block) = items.each(&block)
          def map(&block) = items.map(&block)
          def length = items.length
          def size = items.size
          def empty? = items.empty?
          def first = items.first
          def last = items.last
          def to_a = items
        end

      def initialize(model, config)
        @tasks = build_tasks(model, config)
      end

      def [](text, **options)
        @tasks.each_with_object({}) do |(name, task), out|
          out[name] = Result.new(name: name, items: task.call(text, **options))
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
