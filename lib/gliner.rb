# frozen_string_literal: true

require 'gliner/version'
require 'gliner/configuration'
require 'gliner/model'
require 'gliner/runners/prepared_task'
require 'gliner/runners/entity_runner'
require 'gliner/runners/structured_runner'
require 'gliner/runners/classification_runner'

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

  FormatOptions = Data.define(:include_confidence, :include_spans) do
    def self.from(input)
      return input if input.is_a?(FormatOptions)

      new(
        include_confidence: input.fetch(:include_confidence, false),
        include_spans: input.fetch(:include_spans, false)
      )
    end
  end

  class << self
    attr_writer :model
    attr_writer :config

    def configure
      yield(config)
    end

    def config
      @config ||= Configuration.new
    end

    def load(dir, file: 'model_int8.onnx')
      self.model = Model.from_dir(dir, file: file)
    end

    def model
      @model ||= model_from_env
    end

    def model!
      fetch_model!
    end

    def [](config)
      runner_for(config).new(fetch_model!, config)
    end

    def classify
      Runners::ClassificationRunner
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

    def runner_for(config)
      return Runners::StructuredRunner if structured_config?(config)

      Runners::EntityRunner
    end

    def structured_config?(config)
      return false unless config.is_a?(Hash)

      keys = config.transform_keys(&:to_s)
      return true if keys.key?('name') && keys.key?('fields')

      config.values.all? { |value| value.is_a?(Array) }
    end
  end
end
