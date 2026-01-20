# frozen_string_literal: true

module Gliner
  module Runners
    module Inspectable
      attr_reader :config

      def inspect
        "#<Gliner(#{inspect_label}) config=#{config.inspect}>"
      end
    end
  end
end
