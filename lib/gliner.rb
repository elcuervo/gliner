# frozen_string_literal: true

require 'fileutils'
require 'httpx'
require 'uri'

require 'gliner/version'
require 'gliner/configuration'
require 'gliner/model'
require 'gliner/runners/prepared_task'
require 'gliner/runners/entity_runner'
require 'gliner/runners/structured_runner'
require 'gliner/runners/classification_runner'

module Gliner
  DEFAULT_MODEL_SOURCE = 'https://huggingface.co/cuerbot/gliner2-multi-v1/tree/main/onnx'

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
      @model ||= model_from_config || model_from_env
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

    def model_from_config
      source = config.model
      return nil if source.nil? || source.empty?
      return nil if url?(source)

      file = model_file_for_variant(config.variant)
      Model.from_dir(source, file: file)
    end

    def model_from_env
      dir = ENV.fetch('GLINER_MODEL_DIR', nil)
      return nil if dir.nil? || dir.empty?

      file = ENV['GLINER_MODEL_FILE']
      file = nil if file&.empty?
      file ||= model_file_for_variant(config.variant)
      Model.from_dir(dir, file: file)
    end

    def fetch_model!
      model = self.model
      return model if model

      raise Error, 'No model loaded. Call Gliner.load("/path/to/model"), set config.model, or set GLINER_MODEL_DIR.'
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

    def reset_model!
      @model = nil
    end

    def apply_model_source!
      return unless config.model || config.auto?

      source = config.model
      if config.auto?
        source = DEFAULT_MODEL_SOURCE if source.nil? || source.empty?
        config.model = ensure_model_dir!(source)
      else
        config.model = local_model_dir_from(source) || source
      end
    end

    def local_model_dir_from(source)
      return nil if source.nil? || source.empty?
      return nil if url?(source)

      source
    end

    def ensure_model_dir!(source)
      return source unless url?(source)

      download_model(source)
    end

    def download_model(source)
      repo_id, subdir = parse_hf_source(source)
      raise Error, "Unsupported model source: #{source.inspect}" if repo_id.nil?

      model_file = model_file_for_variant(config.variant)
      dir = File.expand_path("../tmp/models/#{repo_id.tr('/', '__')}", __dir__)
      FileUtils.mkdir_p(dir)

      files = ['tokenizer.json', 'config.json', model_file]
      files.each do |file|
        download_file(hf_resolve_url(repo_id, subdir, file), File.join(dir, file))
      end

      dir
    end

    def parse_hf_source(source)
      return [nil, nil] unless url?(source)

      uri = URI(source)
      return [nil, nil] unless uri.host&.include?('huggingface.co')

      segments = uri.path.split('/').reject(&:empty?)
      return [nil, nil] if segments.length < 2

      repo_id = segments[0, 2].join('/')
      subdir =
        if segments.length > 2 && %w[tree blob resolve].include?(segments[2])
          segments[4..]&.join('/')
        elsif segments.length > 2
          segments[2..]&.join('/')
        end
      subdir = nil if subdir.nil? || subdir.empty?

      [repo_id, subdir]
    rescue URI::InvalidURIError
      [nil, nil]
    end

    def hf_resolve_url(repo_id, subdir, filename)
      base = "https://huggingface.co/#{repo_id}/resolve/main"
      base = "#{base}/#{subdir}" if subdir && !subdir.empty?
      "#{base}/#{filename}"
    end

    def download_file(url, dest, limit: 5)
      return if File.exist?(dest) && File.size?(dest)
      raise Error, "Too many redirects while fetching #{url}" if limit <= 0

      client = HTTPX.plugin(:follow_redirects).with(max_redirects: limit)
      response = client.get(url)
      unless response.status.between?(200, 299)
        raise Error, "Download failed: #{url} (status: #{response.status})"
      end

      File.binwrite(dest, response.body.to_s)
    end

    def url?(value)
      value.is_a?(String) && value.match?(/\Ahttps?:\/\//)
    end

    def model_file_for_variant(variant = :fp16)
      case variant
      when :fp16
        'model_fp16.onnx'
      when :fp32
        'model.onnx'
      when :int8
        'model_int8.onnx'
      else
        raise Error, "Unknown model variant: #{variant.inspect}"
      end
    end
  end
end
