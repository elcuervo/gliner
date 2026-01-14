# frozen_string_literal: true

require 'spec_helper'

describe Gliner::TextProcessor do
  let(:tokenizer) { instance_double('Tokenizers::Tokenizer') }
  let(:processor) { described_class.new(tokenizer) }

  describe '#normalize_text' do
    it 'adds period to text without punctuation' do
      expect(processor.normalize_text('Hello world')).to eq('Hello world.')
    end

    it 'keeps text with existing punctuation' do
      expect(processor.normalize_text('Hello world!')).to eq('Hello world!')
      expect(processor.normalize_text('What?')).to eq('What?')
    end

    it 'handles empty text' do
      expect(processor.normalize_text('')).to eq('.')
    end
  end
end
