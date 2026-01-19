# frozen_string_literal: true

module Gliner
  FormatOptions = Data.define(:include_confidence, :include_spans) do
    def self.from(hash)
      new(
        include_confidence: hash.fetch(:include_confidence, false),
        include_spans: hash.fetch(:include_spans, false)
      )
    end
  end
end
