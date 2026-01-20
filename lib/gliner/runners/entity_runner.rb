# frozen_string_literal: true

module Gliner
  module Runners
    class Entity
      include Inspectable

      def initialize(model, config)
        parsed = model.entity_task.parse_config(config)

        @labels = parsed[:labels]
        @task = PreparedTask.new(model.entity_task, parsed)
      end

      def [](text, **options)
        result = @task.call(text, **options)
        result.fetch('entities')
      end

      alias call []

      private

      def inspect_label = 'Entity'
      def inspect_items = @labels
    end

  end
end
