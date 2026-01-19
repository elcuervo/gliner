# frozen_string_literal: true

module Gliner
  module Runners
    class ClassificationRunner
      def self.[](tasks)
        new(Gliner.model!, tasks)
      end

      def initialize(model, tasks_config)
        raise Error, 'tasks must be a Hash' unless tasks_config.is_a?(Hash)

        @tasks = tasks_config.map do |name, config|
          parsed = model.classification_task.parse_config(name: name, config: config)
          [name.to_s, PreparedTask.new(model.classification_task, parsed)]
        end
      end

      def [](text, **options)
        @tasks.each_with_object({}) do |(name, task), out|
          out[name] = task.call(text, **options)
        end
      end

      alias call []
    end

  end
end
