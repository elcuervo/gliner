# frozen_string_literal: true

require 'spec_helper'
require 'json'

RSpec.describe 'Python compatibility', if: ENV.key?('GLINER_INTEGRATION') do
  let(:fixtures_path) { File.join(__dir__, 'fixtures', 'python_compat.json') }
  let(:fixtures) { JSON.parse(File.read(fixtures_path)) }
  let(:model_file) { ENV.fetch('GLINER_MODEL_FILE', fixtures.dig('meta', 'model_file') || 'model.onnx') }
  let(:model_dir) { ENV['GLINER_MODEL_DIR'] || existing_fixture_dir || raise('Missing GLINER_MODEL_DIR or fixture model_dir') }
  let(:model) { Gliner.load(model_dir, file: model_file) }

  before { model }

  def entity_types_for(input)
    labels = input.fetch('labels')
    descriptions = input.fetch('descriptions', {})

    return labels if descriptions.nil? || descriptions.empty?

    labels.each_with_object({}) do |label, config|
      desc = descriptions[label]
      config[label] = desc.nil? ? nil : desc
    end
  end

  it 'matches Python entities extraction cases' do
    fixtures.fetch('entities').each_with_index do |case_data, idx|
      input = case_data.fetch('input')
      expected = case_data.fetch('expected').fetch('entities')

      result = Gliner[entity_types_for(input)][input.fetch('text'), threshold: input.fetch('threshold')]

      expect(result).to eq(expected), "entities case ##{idx} mismatch"
    end
  end

  it 'matches Python classification cases' do
    fixtures.fetch('classification').each_with_index do |case_data, idx|
      input = case_data.fetch('input')
      expected = case_data.fetch('expected')

      task_name = input.fetch('task_name')
      task_config = {
        'labels' => input.fetch('labels'),
        'multi_label' => input.fetch('multi_label'),
        'cls_threshold' => input.fetch('threshold')
      }

      result = Gliner.classify[{ task_name => task_config }][input.fetch('text')]

      expect(result).to eq(expected), "classification case ##{idx} mismatch"
    end
  end

  it 'matches Python structured extraction cases' do
    fixtures.fetch('json').each_with_index do |case_data, idx|
      input = case_data.fetch('input')
      expected = case_data.fetch('expected')

      result = Gliner[{ input.fetch('parent') => input.fetch('fields') }][
        input.fetch('text'),
        threshold: input.fetch('threshold')
      ]

      expect(result).to eq(expected), "json case ##{idx} mismatch"
    end
  end

  def existing_fixture_dir
    dir = fixtures.dig('meta', 'model_dir')
    return unless dir && Dir.exist?(dir)
    return dir if File.exist?(File.join(dir, model_file))
  end

end
