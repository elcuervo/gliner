# frozen_string_literal: true

require 'spec_helper'

describe Gliner do
  describe '.model' do
    it 'uses config.model_file when set' do
      config = Gliner.config
      previous_model = Gliner.instance_variable_get(:@model)
      previous_dir = config.model_dir
      previous_file = config.model_file

      Gliner.instance_variable_set(:@model, nil)
      config.model_dir = '/tmp/model'
      config.model_file = 'custom.onnx'

      expect(Gliner::Model).to receive(:from_dir).with('/tmp/model', file: 'custom.onnx').and_return(:model)

      expect(Gliner.model).to eq(:model)
    ensure
      Gliner.instance_variable_set(:@model, previous_model)
      config.model_dir = previous_dir
      config.model_file = previous_file
    end
  end
end
