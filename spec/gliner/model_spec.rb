# frozen_string_literal: true

require 'spec_helper'

describe Gliner::Model do
  describe '.from_dir' do
    it 'raises error when files are missing' do
      expect { described_class.from_dir('does-not-exist') }
        .to raise_error(Gliner::Error)
    end
  end

  describe '#format_spans' do
    it 'filters overlapping spans correctly' do
      model = allocate_model_for_unit_test

      spans = [
        ['Tim', 0.8, 0, 3],
        ['Tim Cook', 0.9, 0, 8],
        ['Cook', 0.7, 4, 8]
      ]

      out = model.send(:format_spans, spans, include_confidence: false, include_spans: false)
      expect(out).to eq(['Tim Cook'])
    end
  end

  describe '#parse_field_spec' do
    it 'parses choices, dtype, and description' do
      model = allocate_model_for_unit_test

      parsed = model.send(:parse_field_spec, 'party_size::[1|2|3+]::str::Number of guests')

      expect(parsed[:name]).to eq('party_size')
      expect(parsed[:dtype]).to eq(:str)
      expect(parsed[:choices]).to eq(['1', '2', '3+'])
      expect(parsed[:description]).to eq('Number of guests')
    end
  end

  describe '#filter_spans_by_choices' do
    it 'filters to matching choices when available' do
      model = allocate_model_for_unit_test
      spans = [
        ['outdoor', 0.9, 10, 17],
        ['patio', 0.8, 20, 25]
      ]

      filtered = model.send(:filter_spans_by_choices, spans, ['indoor', 'outdoor'])

      expect(filtered.map(&:first)).to eq(['outdoor'])
    end

    it 'keeps original spans if no choices match' do
      model = allocate_model_for_unit_test
      spans = [
        ['outdoor', 0.9, 10, 17],
        ['patio', 0.8, 20, 25]
      ]

      filtered = model.send(:filter_spans_by_choices, spans, ['inside'])

      expect(filtered).to eq(spans)
    end
  end

  describe '#build_structure_instances' do
    it 'groups spans into multiple instances based on anchor field' do
      model = allocate_model_for_unit_test

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

      instances = model.send(:build_structure_instances, parsed_fields, spans_by_label,
                             include_confidence: false, include_spans: false)

      expect(instances.length).to eq(2)
      expect(instances[0]).to eq({ 'date' => 'Jan 5', 'merchant' => 'Starbucks', 'amount' => '$5.50' })
      expect(instances[1]).to eq({ 'date' => 'Jan 6', 'merchant' => 'Amazon', 'amount' => '$156.99' })
    end
  end

  def allocate_model_for_unit_test
    model = described_class.allocate
    model.instance_variable_set(:@max_width, 8)
    model
  end
end
