# frozen_string_literal: true

module Gliner
  module Runners
    class EntityRunner
      def initialize(model, config)
        parsed = model.entity_task.parse_config(config)
        @task = PreparedTask.new(model.entity_task, parsed)
      end

      def [](text, **options)
        result = @task.call(text, **options)
        result.fetch('entities')
      end

      alias call []
    end
  end
end
