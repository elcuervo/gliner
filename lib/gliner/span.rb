# frozen_string_literal: true

module Gliner
  Span = Data.define(:text, :score, :start, :end) do
    def overlaps?(other)
      !(self.end <= other.start || start >= other.end)
    end
  end
end
