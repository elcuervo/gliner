# frozen_string_literal: true

require 'spec_helper'
require 'json'

RSpec.describe 'Python compatibility', if: ENV.key?('GLINER_INTEGRATION') do
  let(:fixtures_path) { File.join(__dir__, 'fixtures', 'python_compat.json') }
  let(:fixtures) { JSON.parse(File.read(fixtures_path)) }

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

      expect(normalize_entities(result)).to eq(expected), "entities case ##{idx} mismatch"
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

      expect(normalize_classification(result)).to eq(normalize_classification(expected)),
        "classification case ##{idx} mismatch"
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

      expect(normalize_entities(result)).to eq(expected), "json case ##{idx} mismatch"
    end
  end

  def existing_fixture_dir
    dir = fixtures.dig('meta', 'model_dir')
    return unless dir && Dir.exist?(dir)
    return dir if File.exist?(File.join(dir, model_file))
  end

  def normalize_entities(value)
    case value
    when Gliner::Entity
      value.text
    when Array
      value.map { |item| normalize_entities(item) }
    when Hash
      value.transform_values { |item| normalize_entities(item) }
    else
      value
    end
  end

  def normalize_classification(value)
    return value.sort if value.is_a?(Array)
    return value.transform_values { |item| normalize_classification(item) } if value.is_a?(Hash)

    value
  end

end
