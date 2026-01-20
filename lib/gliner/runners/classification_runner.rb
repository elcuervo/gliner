# frozen_string_literal: true

module Gliner
  module Runners
    class Classification
      include Inspectable

      def self.[](tasks)
        new(Gliner.model!, tasks)
      end

      def initialize(model, tasks_config)
        raise Error, 'tasks must be a Hash' unless tasks_config.is_a?(Hash)

        @tasks = tasks_config.to_h do |name, config|
          parsed = model.classification_task.parse_config(name: name, config: config)
          [name.to_s, PreparedTask.new(model.classification_task, parsed)]
        end
      end

      def [](text, **options)
        @tasks.transform_values { |task| task.call(text, **options) }
      end

      alias call []

      private

      def inspect_label = 'Classification'
      def inspect_items = @tasks.keys
    end

  end
end
