# frozen_string_literal: true

module Gliner
  class Task
    attr_reader :config_parser, :inference, :input_builder

    def initialize(config_parser:, inference:, input_builder:)
      @config_parser = config_parser
      @inference = inference
      @input_builder = input_builder
    end

    def parse_config(input) = raise NotImplementedError

    def task_type = raise NotImplementedError

    def label_prefix = raise NotImplementedError

    def build_prompt(parsed) = raise NotImplementedError

    def labels(parsed) = raise NotImplementedError

    def process_output(logits, parsed, prepared, options) = raise NotImplementedError

    def normalize_text? = false

    def needs_cls_logits? = false
  end
end
