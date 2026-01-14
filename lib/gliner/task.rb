# frozen_string_literal: true

module Gliner
  # Base class for all GLiNER tasks
  # Defines the interface for task execution
  class Task
    attr_reader :config_parser, :inference, :input_builder

    def initialize(config_parser:, inference:, input_builder:)
      @config_parser = config_parser
      @inference = inference
      @input_builder = input_builder
    end

    # Parse user configuration into normalized format
    # @return [Hash] Normalized configuration
    def parse_config(input)
      raise NotImplementedError, "#{self.class} must implement #parse_config"
    end

    # Get the task type constant for ONNX inference
    # @return [Integer] Task type (TASK_TYPE_ENTITIES, etc.)
    def task_type
      raise NotImplementedError, "#{self.class} must implement #task_type"
    end

    # Get the label prefix for schema tokens
    # @return [String] Label prefix ("[E]", "[L]", or "[C]")
    def label_prefix
      raise NotImplementedError, "#{self.class} must implement #label_prefix"
    end

    # Build the prompt for the model
    # @param parsed [Hash] Parsed configuration
    # @return [String] Prompt text
    def build_prompt(parsed)
      raise NotImplementedError, "#{self.class} must implement #build_prompt"
    end

    # Get labels from parsed configuration
    # @param parsed [Hash] Parsed configuration
    # @return [Array<String>] List of labels
    def labels(parsed)
      raise NotImplementedError, "#{self.class} must implement #labels"
    end

    # Process the model output into final result
    # @param logits Model output logits
    # @param parsed [Hash] Parsed configuration
    # @param prepared [Hash] Prepared inputs
    # @param options [Hash] User options
    # @return Result specific to task type
    def process_output(logits, parsed, prepared, options)
      raise NotImplementedError, "#{self.class} must implement #process_output"
    end

    # Whether this task should normalize text before processing
    # @return [Boolean]
    def normalize_text?
      false
    end

    # Whether this task needs cls_logits output
    # @return [Boolean]
    def needs_cls_logits?
      false
    end
  end
end
