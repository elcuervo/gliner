# frozen_string_literal: true

require 'spec_helper'

describe Gliner::ConfigParser do
  let(:parser) { described_class.new }

  describe '#parse_entity_types' do
    it 'parses array format' do
      result = parser.parse_entity_types(%w[company person])

      expect(result[:labels]).to eq(%w[company person])
      expect(result[:descriptions]).to eq({})
      expect(result[:dtypes]).to eq({})
      expect(result[:thresholds]).to eq({})
    end

    it 'parses hash with descriptions' do
      entities = { 'company' => 'Company or organization names', 'person' => 'Person names' }
      result = parser.parse_entity_types(entities)

      expect(result[:labels]).to eq(%w[company person])
      expect(result[:descriptions]).to eq(entities)
    end

    it 'parses hash with full config' do
      entities = {
        email: {
          description: 'Email addresses',
          dtype: 'str',
          threshold: 0.9
        }
      }

      result = parser.parse_entity_types(entities)

      expect(result[:labels]).to eq(['email'])
      expect(result[:descriptions]).to eq({ 'email' => 'Email addresses' })
      expect(result[:dtypes]).to eq({ 'email' => :str })
      expect(result[:thresholds]).to eq({ 'email' => 0.9 })
    end
  end

  describe '#parse_classification_task' do
    it 'parses simple array format' do
      result = parser.parse_classification_task('sentiment', %w[positive negative])

      expect(result[:labels]).to eq(%w[positive negative])
      expect(result[:multi_label]).to be false
      expect(result[:cls_threshold]).to eq(0.5)
    end

    it 'parses multi-label config' do
      classify = {
        labels: %w[camera battery screen],
        multi_label: true,
        cls_threshold: 0.4
      }

      result = parser.parse_classification_task('aspects', classify)

      expect(result[:labels]).to eq(%w[camera battery screen])
      expect(result[:multi_label]).to be true
      expect(result[:cls_threshold]).to eq(0.4)
    end
  end

  describe '#build_prompt' do
    it 'builds prompt with descriptions' do
      entities = {
        company: 'Organization names',
        person: 'Full names of people'
      }

      prompt = parser.build_prompt('entities', entities)

      expect(prompt).to include('entities')
      expect(prompt).to include('[DESCRIPTION] company: Organization names')
      expect(prompt).to include('[DESCRIPTION] person: Full names of people')
    end

    it 'handles empty descriptions' do
      prompt = parser.build_prompt('entities', {})
      expect(prompt).to eq('entities')
    end
  end
end
