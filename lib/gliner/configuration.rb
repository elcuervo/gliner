# frozen_string_literal: true

module Gliner
  class Configuration
    DEFAULT_THRESHOLD = 0.5

    attr_accessor :threshold, :model
    attr_reader :variant

    def initialize
      @threshold = DEFAULT_THRESHOLD
      @model = nil
      @variant = :fp16
      @auto = false
    end

    def variant=(value)
      @variant = value.nil? ? nil : value.to_sym
    end

    def auto!(value = true)
      @auto = !!value
    end

    def auto=(value)
      @auto = !!value
    end

    def auto?
      @auto
    end
  end
end
