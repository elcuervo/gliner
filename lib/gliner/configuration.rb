# frozen_string_literal: true

module Gliner
  class Configuration
    DEFAULT_THRESHOLD = 0.5

    attr_accessor :threshold, :model_dir

    def initialize
      @threshold = DEFAULT_THRESHOLD
      @model_dir = nil
    end
  end
end
