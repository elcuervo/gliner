# frozen_string_literal: true

require 'spec_helper'
require 'tmpdir'

describe Gliner do
  describe '.model' do
    before do
      @previous_model = Gliner.instance_variable_get(:@model)
      @previous_config = Gliner.instance_variable_get(:@config)
      Gliner.instance_variable_set(:@model, nil)
      Gliner.instance_variable_set(:@config, Gliner::Configuration.new)
    end

    after do
      Gliner.instance_variable_set(:@model, @previous_model)
      Gliner.instance_variable_set(:@config, @previous_config)
    end

    it 'uses config.model when set' do
      config = Gliner.config

      config.model = '/tmp/model'
      config.variant = :fp32

      expect(Gliner::Model).to receive(:from_dir).with('/tmp/model', file: 'model.onnx').and_return(:model)

      expect(Gliner.model).to eq(:model)
    end

    it 'defaults to int8 when no variant is set' do
      config = Gliner.config
      config.model = '/tmp/model'

      expect(Gliner::Model).to receive(:from_dir).with('/tmp/model', file: 'model_int8.onnx').and_return(:model)

      expect(Gliner.model).to eq(:model)
    end

    it 'uses variant to choose the model file' do
      config = Gliner.config
      config.model = '/tmp/model'
      config.variant = :int8

      expect(Gliner::Model).to receive(:from_dir).with('/tmp/model', file: 'model_int8.onnx').and_return(:model)

      expect(Gliner.model).to eq(:model)
    end

    it 'auto! downloads when model is not set' do
      allow(Gliner).to receive(:download_default_model).and_return('/tmp/model')

      Gliner.configure do |config|
        config.variant = :fp16
        config.auto!
      end

      expect(Gliner.config.model).to eq('/tmp/model')
      expect(Gliner).to have_received(:download_default_model)
    end

    it 'auto! uses a local path when provided' do
      Dir.mktmpdir do |dir|
        allow(Gliner).to receive(:download_default_model)

        Gliner.configure do |config|
          config.model = dir
          config.auto!
        end

        expect(Gliner.config.model).to eq(dir)
        expect(Gliner).not_to have_received(:download_default_model)
      end
    end

    it 'auto! does not download when model is explicitly set' do
      allow(Gliner).to receive(:download_default_model)

      Gliner.configure do |config|
        config.model = ''
        config.auto!
      end

      expect(Gliner.config.model).to eq('')
      expect(Gliner).not_to have_received(:download_default_model)
    end

    it 'auto! respects GLINER_MODEL_DIR when set' do
      allow(Gliner).to receive(:download_default_model)

      previous = ENV['GLINER_MODEL_DIR']
      ENV['GLINER_MODEL_DIR'] = '/tmp/env-model'

      begin
        Gliner.configure do |config|
          config.auto!
        end

        expect(Gliner.config.model).to be_nil
        expect(Gliner).not_to have_received(:download_default_model)
      ensure
        ENV['GLINER_MODEL_DIR'] = previous
      end
    end
  end
end
