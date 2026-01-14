# frozen_string_literal: true

require 'json'
require 'onnxruntime'
require 'tokenizers'

require_relative 'span'
require_relative 'prepared_input'
require_relative 'text_processor'
require_relative 'config_parser'
require_relative 'inference'
require_relative 'input_builder'
require_relative 'span_extractor'
require_relative 'classifier'
require_relative 'structured_extractor'
require_relative 'task'
require_relative 'pipeline'
require_relative 'tasks/entity_extraction'
require_relative 'tasks/classification'
require_relative 'tasks/json_extraction'

module Gliner
  class Model
    DEFAULT_MAX_WIDTH = 8
    DEFAULT_MAX_SEQ_LEN = 512

    def self.from_dir(dir, file: 'model_int8.onnx')
      raise Error, "Model directory not found: #{dir}" unless Dir.exist?(dir)

      config_path = File.join(dir, 'config.json')
      config = File.exist?(config_path) ? JSON.parse(File.read(config_path)) : {}

      model_path = File.join(dir, file)
      tokenizer_path = File.join(dir, 'tokenizer.json')

      new(
        model_path: model_path,
        tokenizer_path: tokenizer_path,
        max_width: config.fetch('max_width', DEFAULT_MAX_WIDTH),
        max_seq_len: config.fetch('max_seq_len', DEFAULT_MAX_SEQ_LEN)
      )
    end

    def initialize(model_path:, tokenizer_path:, max_width: DEFAULT_MAX_WIDTH, max_seq_len: DEFAULT_MAX_SEQ_LEN)
      @model_path = model_path
      @tokenizer_path = tokenizer_path
      @max_width = Integer(max_width)
      @max_seq_len = Integer(max_seq_len)

      raise Error, "Model file not found: #{@model_path}" unless File.exist?(@model_path)
      raise Error, "Tokenizer file not found: #{@tokenizer_path}" unless File.exist?(@tokenizer_path)

      tokenizer = Tokenizers.from_file(@tokenizer_path)
      session = OnnxRuntime::InferenceSession.new(@model_path)

      @text_processor = TextProcessor.new(tokenizer)
      @inference = Inference.new(session)
      @input_builder = InputBuilder.new(@text_processor, max_seq_len: @max_seq_len)
      @span_extractor = SpanExtractor.new(@inference, max_width: @max_width)
      @classifier = Classifier.new(@inference, max_width: @max_width)
      @structured_extractor = StructuredExtractor.new(@span_extractor)
    end

    def config_parser
      @config_parser ||= ConfigParser.new
    end

    def pipeline
      @pipeline ||= Pipeline.new(text_processor: @text_processor, inference: @inference)
    end

    def entity_task
      @entity_task ||= Tasks::EntityExtraction.new(
        config_parser: config_parser,
        inference: @inference,
        input_builder: @input_builder,
        span_extractor: @span_extractor
      )
    end

    def classification_task
      @classification_task ||= Tasks::Classification.new(
        config_parser: config_parser,
        inference: @inference,
        input_builder: @input_builder,
        classifier: @classifier
      )
    end

    def json_task
      @json_task ||= Tasks::JsonExtraction.new(
        config_parser: config_parser,
        inference: @inference,
        input_builder: @input_builder,
        span_extractor: @span_extractor,
        structured_extractor: @structured_extractor
      )
    end

    # Extract named entities from text
    #
    # Supports:
    # - Array: ["company", "person"]
    # - Hash: {"company"=>"desc", "person"=>"desc"}
    # - Hash config: {"email"=>{"description"=>"...", "dtype"=>"str", "threshold"=>0.9}}
    #
    # Returns: {"entities" => {"label" => ["span", ...], ...}}
    #
    # @param text [String] Input text
    # @param entity_types [Array, Hash] Entity types configuration
    # @param threshold [Float, nil] Optional override for task cls_threshold
    # @param include_confidence [Boolean] Include confidence scores (default: false)
    # @param include_spans [Boolean] Include character positions (default: false)
    # @return [Hash] Extracted entities
    #
    def extract_entities(text, entity_types, threshold: 0.5, format_results: true, include_confidence: false, include_spans: false)
      pipeline.execute(
        entity_task,
        text,
        entity_types,
        threshold: threshold,
        include_confidence: include_confidence,
        include_spans: include_spans
      )
    end

    # Classify text into one or more categories
    #
    # Supports:
    # - {"sentiment" => ["positive","negative","neutral"]}
    # - {"aspects" => {"labels"=>[...], "multi_label"=>true, "cls_threshold"=>0.4}}
    # - {"sentiment" => {"labels"=>{"positive"=>"desc", ...}}}
    #
    # Returns:
    # - Single-label: {"sentiment"=>"negative"}
    # - Multi-label: {"aspects"=>["camera","price"]}
    #
    # @param text [String] Input text
    # @param tasks [Hash] Classification tasks configuration
    # @param threshold [Float] Confidence threshold (default: 0.5)
    # @param include_confidence [Boolean] Include confidence scores (default: false)
    # @return [Hash] Classification results
    #
    def classify_text(text, tasks, threshold: nil, format_results: true, include_confidence: false, include_spans: false)
      options = { include_confidence: include_confidence }
      options[:threshold] = threshold unless threshold.nil?

      classification_task.execute_all(pipeline, text, tasks, **options)
    end

    # Extract structured data from text
    #
    # Supports field specs like:
    #   "name::str::Full product name and model"
    #   "category::[food|transport|shopping]::str"
    #
    # Returns:
    #   {"product"=>[{"name"=>"...", ...}]}
    #
    # @param text [String] Input text
    # @param structures [Hash] Structure definitions
    # @param threshold [Float] Confidence threshold (default: 0.5)
    # @param include_confidence [Boolean] Include confidence scores (default: false)
    # @param include_spans [Boolean] Include character positions (default: false)
    # @return [Hash] Extracted structured data
    #
    def extract_json(text, structures, threshold: 0.5, format_results: true, include_confidence: false, include_spans: false)
      json_task.execute_all(
        pipeline,
        text,
        structures,
        threshold: threshold,
        include_confidence: include_confidence,
        include_spans: include_spans
      )
    end
  end
end
