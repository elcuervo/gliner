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

  def allocate_model_for_unit_test
    model = described_class.allocate
    model.instance_variable_set(:@max_width, 8)
    model
  end
end
