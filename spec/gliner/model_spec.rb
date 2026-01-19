# frozen_string_literal: true

require 'spec_helper'

describe Gliner::SpanExtractor do
  describe '#format_spans' do
    it 'filters overlapping spans correctly' do
      inference = double('Inference')
      span_extractor = described_class.new(inference, max_width: 8)

      spans = [
        Gliner::Span.new(text: 'Tim', score: 0.8, start: 0, end: 3),
        Gliner::Span.new(text: 'Tim Cook', score: 0.9, start: 0, end: 8),
        Gliner::Span.new(text: 'Cook', score: 0.7, start: 4, end: 8)
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
        Gliner::Span.new(text: 'outdoor', score: 0.9, start: 10, end: 17),
        Gliner::Span.new(text: 'patio', score: 0.8, start: 20, end: 25)
      ]

      filtered = structured_extractor.filter_spans_by_choices(spans, %w[indoor outdoor])

      expect(filtered.map(&:text)).to eq(['outdoor'])
    end

    it 'keeps original spans if no choices match' do
      spans = [
        Gliner::Span.new(text: 'outdoor', score: 0.9, start: 10, end: 17),
        Gliner::Span.new(text: 'patio', score: 0.8, start: 20, end: 25)
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
        'date' => [
          Gliner::Span.new(text: 'Jan 5', score: 0.9, start: 0, end: 5),
          Gliner::Span.new(text: 'Jan 6', score: 0.8, start: 40, end: 45)
        ],

        'merchant' => [
          Gliner::Span.new(text: 'Starbucks', score: 0.9, start: 7, end: 16),
          Gliner::Span.new(text: 'Amazon', score: 0.7, start: 47, end: 53)
        ],

        'amount' => [
          Gliner::Span.new(text: '$5.50', score: 0.9, start: 17, end: 22),
          Gliner::Span.new(text: '$156.99', score: 0.8, start: 54, end: 61)
        ]
      }

      instances = structured_extractor
        .build_structure_instances(parsed_fields, spans_by_label, include_confidence: false, include_spans: false)

      expect(instances.length).to eq(2)
      expect(instances[0]).to eq({ 'date' => 'Jan 5', 'merchant' => 'Starbucks', 'amount' => '$5.50' })
      expect(instances[1]).to eq({ 'date' => 'Jan 6', 'merchant' => 'Amazon', 'amount' => '$156.99' })
    end
  end
end
