# frozen_string_literal: true

module Gliner
  module Runners
    module Inspectable
      def inspect
        items = Array(inspect_items).map(&:to_s)

        "#<Gliner(#{inspect_label}) input=#{items.inspect}>"
      end
    end
  end
end
