# frozen_string_literal: true

require 'spec_helper'
require 'json'
require 'fileutils'

RSpec.describe 'Python compatibility', if: ENV.key?('GLINER_INTEGRATION') do
  let(:fixtures_path) { File.join(__dir__, 'fixtures', 'python_compat.json') }
  let(:fixtures) { JSON.parse(File.read(fixtures_path)) }
  let(:repo_id) { ENV.fetch('GLINER_REPO_ID', fixtures.dig('meta', 'repo_id') || 'fastino/gliner2-multi-v1') }
  let(:model_file) { ENV.fetch('GLINER_MODEL_FILE', fixtures.dig('meta', 'model_file') || 'model.onnx') }
  let(:model_dir) { ENV['GLINER_MODEL_DIR'] || existing_fixture_dir || ensure_model_dir!(repo_id, model_file) }
  let(:model) { Gliner::Model.from_dir(model_dir, file: model_file) }

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
      expected = case_data.fetch('expected')

      result = model.extract_entities(
        input.fetch('text'),
        entity_types_for(input),
        threshold: input.fetch('threshold')
      )

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

      result = model.classify_text(input.fetch('text'), { task_name => task_config })

      expect(result).to eq(expected), "classification case ##{idx} mismatch"
    end
  end

  it 'matches Python structured extraction cases' do
    fixtures.fetch('json').each_with_index do |case_data, idx|
      input = case_data.fetch('input')
      expected = case_data.fetch('expected')

      result = model.extract_json(
        input.fetch('text'),
        { input.fetch('parent') => input.fetch('fields') },
        threshold: input.fetch('threshold')
      )

      expect(result).to eq(expected), "json case ##{idx} mismatch"
    end
  end

  def existing_fixture_dir
    dir = fixtures.dig('meta', 'model_dir')
    return unless dir && Dir.exist?(dir)
    return dir if File.exist?(File.join(dir, model_file))
  end

  def ensure_model_dir!(repo_id, model_file)
    dir = File.expand_path("../.context/onnx/#{repo_id.tr('/', '__')}", __dir__)
    model_path = File.join(dir, model_file)
    return dir if File.exist?(model_path)

    FileUtils.mkdir_p(dir)
    export_onnx(repo_id, dir, model_file)
    dir
  end

  def export_onnx(repo_id, dir, model_file)
    python = File.expand_path('../.context/pyenv/bin/python', __dir__)
    unless File.exist?(python)
      raise "Python venv not found at #{python}. Run the fixture generator to create it."
    end

    args = [
      python,
      File.expand_path('../onnx/export.py', __dir__),
      '--model-id', repo_id,
      '--output-dir', dir,
      '--no-validate',
      '--no-validate-extraction'
    ]
    args << '--no-quantize' unless model_file == 'model_int8.onnx'

    system(*args) or raise "ONNX export failed for #{repo_id}"
  end
end
