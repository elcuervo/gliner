# frozen_string_literal: true

require 'spec_helper'

describe Gliner::Model do
  describe '.from_dir' do
    it 'raises error when files are missing' do
      expect { described_class.from_dir('does-not-exist') }
        .to raise_error(Gliner::Error)
    end
  end
end

describe Gliner::SpanExtractor do
  describe '#format_spans' do
    it 'filters overlapping spans correctly' do
      inference = double('Inference')
      span_extractor = described_class.new(inference, max_width: 8)

      spans = [
        ['Tim', 0.8, 0, 3],
        ['Tim Cook', 0.9, 0, 8],
        ['Cook', 0.7, 4, 8]
      ]

      out = span_extractor.format_spans(spans, include_confidence: false, include_spans: false)
      expect(out).to eq(['Tim Cook'])
    end
  end
end

describe Gliner::ConfigParser do
  describe '#parse_field_spec' do
    it 'parses choices, dtype, and description' do
      parser = described_class.new

      parsed = parser.parse_field_spec('party_size::[1|2|3+]::str::Number of guests')

      expect(parsed[:name]).to eq('party_size')
      expect(parsed[:dtype]).to eq(:str)
      expect(parsed[:choices]).to eq(['1', '2', '3+'])
      expect(parsed[:description]).to eq('Number of guests')
    end
  end
end

describe Gliner::StructuredExtractor do
  let(:inference) { double('Inference') }
  let(:span_extractor) { Gliner::SpanExtractor.new(inference, max_width: 8) }
  let(:structured_extractor) { described_class.new(span_extractor) }

  describe '#filter_spans_by_choices' do
    it 'filters to matching choices when available' do
      spans = [
        ['outdoor', 0.9, 10, 17],
        ['patio', 0.8, 20, 25]
      ]

      filtered = structured_extractor.filter_spans_by_choices(spans, ['indoor', 'outdoor'])

      expect(filtered.map(&:first)).to eq(['outdoor'])
    end

    it 'keeps original spans if no choices match' do
      spans = [
        ['outdoor', 0.9, 10, 17],
        ['patio', 0.8, 20, 25]
      ]

      filtered = structured_extractor.filter_spans_by_choices(spans, ['inside'])

      expect(filtered).to eq(spans)
    end
  end

  describe '#build_structure_instances' do
    it 'groups spans into multiple instances based on anchor field' do
      parsed_fields = [
        { name: 'date', dtype: :str },
        { name: 'merchant', dtype: :str },
        { name: 'amount', dtype: :str }
      ]

      spans_by_label = {
        'date' => [['Jan 5', 0.9, 0, 5], ['Jan 6', 0.8, 40, 45]],
        'merchant' => [['Starbucks', 0.9, 7, 16], ['Amazon', 0.7, 47, 53]],
        'amount' => [['$5.50', 0.9, 17, 22], ['$156.99', 0.8, 54, 61]]
      }

      instances = structured_extractor.build_structure_instances(parsed_fields, spans_by_label,
                                                                  include_confidence: false, include_spans: false)

      expect(instances.length).to eq(2)
      expect(instances[0]).to eq({ 'date' => 'Jan 5', 'merchant' => 'Starbucks', 'amount' => '$5.50' })
      expect(instances[1]).to eq({ 'date' => 'Jan 6', 'merchant' => 'Amazon', 'amount' => '$156.99' })
    end
  end
end
