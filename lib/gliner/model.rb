# frozen_string_literal: true

require 'json'
require 'onnxruntime'
require 'tokenizers'

require 'gliner/text_processor'
require 'gliner/config_parser'
require 'gliner/inference'
require 'gliner/input_builder'
require 'gliner/span_extractor'
require 'gliner/classifier'
require 'gliner/structured_extractor'
require 'gliner/task'
require 'gliner/pipeline'
require 'gliner/tasks/entity_extraction'
require 'gliner/tasks/classification'
require 'gliner/tasks/json_extraction'

module Gliner
  class Model
    DEFAULT_MAX_WIDTH = 8
    DEFAULT_MAX_SEQ_LEN = 512

    def self.from_dir(dir, file: 'model_int8.onnx')
      config_path = File.join(dir, 'config.json')
      config = File.exist?(config_path) ? JSON.parse(File.read(config_path)) : {}

      new(
        model_path: File.join(dir, file),
        tokenizer_path: File.join(dir, 'tokenizer.json'),
        max_width: config.fetch('max_width', DEFAULT_MAX_WIDTH),
        max_seq_len: config.fetch('max_seq_len', DEFAULT_MAX_SEQ_LEN)
      )
    end

    def initialize(model_path:, tokenizer_path:, max_width: DEFAULT_MAX_WIDTH, max_seq_len: DEFAULT_MAX_SEQ_LEN)
      @model_path = model_path
      @tokenizer_path = tokenizer_path
      @max_width = Integer(max_width)
      @max_seq_len = Integer(max_seq_len)

      tokenizer = Tokenizers.from_file(@tokenizer_path)
      session = OnnxRuntime::InferenceSession.new(@model_path)

      @text_processor = TextProcessor.new(tokenizer)
      @inference = Inference.new(session)
    end

    def config_parser
      @config_parser ||= ConfigParser.new
    end

    def pipeline
      @pipeline ||= Pipeline.new(text_processor: @text_processor, inference: @inference)
    end

    def input_builder
      @input_builder ||= InputBuilder.new(@text_processor, max_seq_len: @max_seq_len)
    end

    def span_extractor
      @span_extractor ||= SpanExtractor.new(@inference, max_width: @max_width)
    end

    def structured_extractor
      @structured_extractor ||= StructuredExtractor.new(span_extractor)
    end

    def classifier
      @classifier ||= Classifier.new(@inference, max_width: @max_width)
    end

    def entity_task
      @entity_task ||= Tasks::EntityExtraction.new(
        config_parser: config_parser,
        inference: @inference,
        input_builder: input_builder,
        span_extractor: span_extractor
      )
    end

    def classification_task
      @classification_task ||= Tasks::Classification.new(
        config_parser: config_parser,
        inference: @inference,
        input_builder: input_builder,
        classifier: classifier
      )
    end

    def json_task
      @json_task ||= Tasks::JsonExtraction.new(
        config_parser: config_parser,
        inference: @inference,
        input_builder: input_builder,
        span_extractor: span_extractor,
        structured_extractor: structured_extractor
      )
    end

    def extract_entities(text, entity_types, **options)
      threshold = options.fetch(:threshold, Gliner.config.threshold)
      include_confidence = options.fetch(:include_confidence, false)
      include_spans = options.fetch(:include_spans, false)

      pipeline.execute(
        entity_task,
        text,
        entity_types,
        threshold: threshold,
        include_confidence: include_confidence,
        include_spans: include_spans
      )
    end

    def classify_text(text, tasks, **options)
      include_confidence = options.fetch(:include_confidence, false)
      threshold = options[:threshold]

      task_options = { include_confidence: include_confidence }
      task_options[:threshold] = threshold unless threshold.nil?

      classification_task.execute_all(pipeline, text, tasks, **task_options)
    end

    def extract_json(text, structures, **options)
      threshold = options.fetch(:threshold, Gliner.config.threshold)
      include_confidence = options.fetch(:include_confidence, false)
      include_spans = options.fetch(:include_spans, false)

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
