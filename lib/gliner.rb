# frozen_string_literal: true

require 'fileutils'
require 'httpx'
require 'gliner/version'
require 'gliner/configuration'
require 'gliner/model'
require 'gliner/runners/prepared_task'
require 'gliner/runners/inspectable'
require 'gliner/runners/entity_runner'
require 'gliner/runners/structured_runner'
require 'gliner/runners/classification_runner'

module Gliner
  HF_REPO = 'cuerbot/gliner2-multi-v1'
  HF_DIR = 'onnx'

  DEFAULT_MODEL_BASE = "https://huggingface.co/#{HF_REPO}/resolve/main/#{HF_DIR}".freeze

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

  Structure = Data.define(:fields) do
    def [](key) = fields[key]
    def fetch(key, *args, &block) = fields.fetch(key, *args, &block)
    def to_h = fields
    def to_hash = fields
    def keys = fields.keys
    def values = fields.values
    def each(&block) = fields.each(&block)
  end

  Entity = Data.define(:index, :offsets, :text, :name, :probability) do
    def to_s = text.to_s
    def to_str = text.to_s
  end

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
    attr_writer :model, :config

    def configure
      yield(config)

      reset_model!
      apply_model_source!
    end

    def config
      @config ||= Configuration.new
    end

    def load(dir, file: nil)
      file ||= ENV['GLINER_MODEL_FILE'] || model_file_for_variant(config.variant)

      self.model = Model.from_dir(dir, file: file)
    end

    def model
      @model ||= begin
        apply_model_source!
        model_from_config || model_from_env
      end
    end

    def [](config)
      runner_for(config).new(model!, config)
    end

    def classify
      Runners::Classification
    end

    def model!
      model = self.model

      return model if model

      raise Error, 'No model loaded. Call Gliner.load("/path/to/model"), set config.model, or set GLINER_MODEL_DIR.'
    end

    private

    def model_from_config
      source = config.model
      return nil if source.nil?

      file = model_file_for_variant(config.variant)
      Model.from_dir(source, file: file)
    end

    def model_from_env
      dir = env_model_dir
      return if dir.nil?

      file = ENV['GLINER_MODEL_FILE'] || model_file_for_variant(config.variant)

      Model.from_dir(dir, file: file)
    end

    def runner_for(config)
      return Runners::Structure if structured_config?(config)

      Runners::Entity
    end

    def structured_config?(config)
      return false unless config.is_a?(Hash)

      keys = config.transform_keys(&:to_s)

      return true if keys.key?('name') && keys.key?('fields')

      config.values.all? { |value| value.is_a?(Array) }
    end

    def reset_model!
      @model = nil
    end

    def apply_model_source!
      return unless config.auto?

      source = config.model

      return unless source.nil?
      return if env_model_dir

      config.model = download_default_model
    end

    def client
      @client ||= HTTPX.plugin(:follow_redirects)
    end

    def download_default_model
      model_file = model_file_for_variant(config.variant)
      dir = File.join(Dir.pwd, '.cache', 'gliner', HF_REPO.tr('/', '__'))

      FileUtils.mkdir_p(dir)

      files = ['tokenizer.json', 'config.json', model_file]

      files.each do |file|
        target = File.join(dir, file)

        next if File.exist?(target) && File.size?(target)

        puts "Downloading #{DEFAULT_MODEL_BASE}/#{file}"

        client
          .get("#{DEFAULT_MODEL_BASE}/#{file}")
          .copy_to(target)
      end

      dir
    end

    def env_model_dir
      dir = ENV.fetch('GLINER_MODEL_DIR', nil)
      return nil if dir.nil? || dir.empty?

      dir
    end

    def model_file_for_variant(variant = :fp16)
      case variant.to_sym
      when :fp16 then 'model_fp16.onnx'
      when :fp32 then 'model.onnx'
      when :int8 then 'model_int8.onnx'
      else
        raise Error, "Unknown model variant: #{variant.inspect}"
      end
    end
  end
end
