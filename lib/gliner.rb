# frozen_string_literal: true

require 'gliner/version'
require 'gliner/model'
require 'gliner/api'

module Gliner
  Error = Class.new(StandardError)

  PreparedInput = Data.define(
    :input_ids,
    :word_ids,
    :attention_mask,
    :words_mask,
    :pos_to_word_index,
    :start_map,
    :end_map,
    :original_text,
    :text_len
  )

  Span = Data.define(:text, :score, :start, :end) do
    def overlaps?(other)
      !(self.end <= other.start || start >= other.end)
    end
  end

  class << self
    def load(dir, file: 'model_int8.onnx')
      self.model = Model.from_dir(dir, file: file)
    end

    def from_dir(dir, file: 'model_int8.onnx')
      load(dir, file: file)
    end

    attr_writer :model

    def model
      @model ||= model_from_env
    end

    def [](config)
      API.compile(fetch_model!, config)
    end

    def classify
      API::ClassificationProxy.new(fetch_model!)
    end

    private

    def model_from_env
      dir = ENV.fetch('GLINER_MODEL_DIR', nil)
      return nil if dir.nil? || dir.empty?

      file = ENV['GLINER_MODEL_FILE'] || 'model_int8.onnx'
      Model.from_dir(dir, file: file)
    end

    def fetch_model!
      model = self.model
      return model if model

      raise Error, 'No model loaded. Call Gliner.load("/path/to/model") or set GLINER_MODEL_DIR.'
    end
  end
end
