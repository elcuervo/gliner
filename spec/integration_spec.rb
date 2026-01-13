# frozen_string_literal: true

require 'spec_helper'
require 'fileutils'

describe 'Gliner Integration', if: ENV.key?('GLINER_INTEGRATION') do
  REPO_ID = 'cuerbot/gliner2-multi-v1'

  describe 'real model inference' do
    context 'with entities extraction' do
      it 'extracts entities correctly' do
        model_dir = ensure_model_dir!
        model = Gliner::Model.from_dir(model_dir)

        text = 'Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.'
        labels = {
          'company' => { 'description' => 'Company or organization names', 'dtype' => 'str' },
          'person' => { 'description' => 'Person names', 'dtype' => 'list' },
          'product' => { 'description' => 'Product names', 'dtype' => 'list', 'threshold' => 0.4 },
          'location' => { 'description' => 'Places' }
        }

        out = model.extract_entities(text, labels, threshold: 0.5)
        entities = out.fetch('entities')

        expect(entities.fetch('company')).to be_a(String)
        expect(entities.fetch('person')).to be_a(Array)
        expect(entities.fetch('company')).to include('Apple')
        expect(entities.fetch('person')).to include('Tim Cook')
        expect(entities.fetch('product')).to include('iPhone 15')
        expect(entities.fetch('location')).to include('Cupertino')
      end
    end

    context 'with text classification' do
      it 'classifies sentiment correctly' do
        model_dir = ensure_model_dir!
        model = Gliner::Model.from_dir(model_dir)

        text = 'This laptop has amazing performance but terrible battery life!'
        schema = { 'sentiment' => %w[positive negative neutral] }

        out = model.classify_text(text, schema)

        expect(out.fetch('sentiment')).to eq('negative')
      end
    end

    context 'with structured extraction' do
      it 'extracts JSON structure correctly' do
        model_dir = ensure_model_dir!
        model = Gliner::Model.from_dir(model_dir)

        text = 'iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199.'
        complex_schema = {
          'product' => [
            'name::str::Full product name and model',
            'storage::str::Storage capacity',
            'processor::str::Chip or processor information',
            'price::str::Product price with currency'
          ]
        }
        out = model.extract_json( text, complex_schema, threshold: 0.4)
        product = out.fetch('product').fetch(0)

        expect(product.fetch('name')).to include('iPhone')
        expect(product.fetch('storage')).to include('256')
        expect(product.fetch('processor')).to include('A17')
        expect(product.fetch('price')).to include('$')
      end

      it 'supports choices and multiple instances' do
        model_dir = ensure_model_dir!
        model = Gliner::Model.from_dir(model_dir)

        text = <<~TEXT
          Transaction 1
          Date: Jan 5
          Merchant: Starbucks
          Amount: $5.50
          Category: food

          Transaction 2
          Date: Jan 6
          Merchant: Amazon
          Amount: $156.99
          Category: shopping

          Order status: shipped
        TEXT

        schema = {
          'transaction' => [
            'date::str',
            'merchant::str',
            'amount::str',
            'category::[food|transport|shopping]::str'
          ],
          'order' => [
            'status::[pending|processing|shipped]::str'
          ]
        }

        out = model.extract_json(text, schema, threshold: 0.2)
        transactions = out.fetch('transaction')
        order = out.fetch('order').fetch(0)

        expect(transactions.length).to be >= 1
        dates = transactions.map { |t| t['date'] }.compact
        expect(dates).not_to be_empty
        expect(dates.join(' ')).to include('Jan')

        categories = transactions.map { |t| t['category'] }.compact
        categories.each do |category|
          expect(%w[food transport shopping]).to include(category)
        end

        expect(%w[pending processing shipped]).to include(order.fetch('status'))
      end
    end
  end

  def ensure_model_dir!
    from_env = ENV.fetch('GLINER_MODEL_DIR', nil)
    return from_env if from_env && !from_env.empty?

    dir = File.expand_path("../tmp/#{REPO_ID.tr('/', '__')}", __dir__)
    FileUtils.mkdir_p(dir)

    download(hf_resolve_url('tokenizer.json').to_s, File.join(dir, 'tokenizer.json'))
    download(hf_resolve_url('config.json').to_s, File.join(dir, 'config.json'))
    download(hf_resolve_url('model_int8.onnx').to_s, File.join(dir, 'model_int8.onnx'))

    dir
  end

  def hf_resolve_url(filename)
    "https://huggingface.co/#{REPO_ID}/resolve/main/#{filename}"
  end

  def download(url, dest)
    return if File.exist?(dest) && File.size?(dest)

    # Use curl to avoid reimplementing HTTP downloads and to support resumes.
    ok = system(
      'curl',
      '--fail',
      '--location',
      '--retry',
      '3',
      '--retry-delay',
      '1',
      '--continue-at',
      '-',
      '--output',
      dest,
      url
    )
    raise "Download failed: #{url}" unless ok
  end
end
