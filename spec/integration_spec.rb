# frozen_string_literal: true

require 'spec_helper'
require 'fileutils'
require 'httpx'

describe 'Gliner Integration', if: ENV.key?('GLINER_INTEGRATION') do
  REPO_ID = 'cuerbot/gliner2-multi-v1'
  REPO_SUBDIR = 'onnx'
  MODEL_FILE = 'model_fp16.onnx'

  describe 'real model inference' do
    context 'with entities extraction' do
      it 'extracts entities correctly' do
        model_dir = ensure_model_dir!
        Gliner.load(model_dir, file: MODEL_FILE)

        text = 'Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.'
        labels = {
          'company' => { 'description' => 'Company or organization names', 'dtype' => 'str' },
          'person' => { 'description' => 'Person names', 'dtype' => 'list' },
          'product' => { 'description' => 'Product names', 'dtype' => 'list', 'threshold' => 0.4 },
          'location' => { 'description' => 'Places' }
        }

        entities = Gliner[labels][text, threshold: 0.5]

        expect(entities.fetch('company')).to be_a(Gliner::Entity)
        expect(entities.fetch('person')).to all(be_a(Gliner::Entity))
        expect(entity_text(entities.fetch('company'))).to include('Apple')
        expect(entity_texts(entities.fetch('person'))).to include('Tim Cook')
        expect(entity_texts(entities.fetch('product'))).to include('iPhone 15')
        expect(entity_texts(entities.fetch('location'))).to include('Cupertino')
      end
    end

    context 'with text classification' do
      it 'classifies sentiment correctly' do
        model_dir = ensure_model_dir!
        Gliner.load(model_dir, file: MODEL_FILE)

        text = 'This laptop has amazing performance but terrible battery life!'
        schema = { 'sentiment' => %w[positive negative neutral] }

        out = Gliner.classify[schema][text]
        sentiment = out.fetch('sentiment')

        expect(sentiment).to be_a(Gliner::Label)
        expect(sentiment.label).to eq('negative')
        expect(sentiment.confidence).to be_a(Float)
      end
    end

    context 'with structured extraction' do
      it 'extracts JSON structure correctly' do
        model_dir = ensure_model_dir!
        Gliner.load(model_dir, file: MODEL_FILE)

        text = 'iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199.'
        complex_schema = {
          'product' => [
            'name::str::Full product name and model',
            'storage::str::Storage capacity',
            'processor::str::Chip or processor information',
            'price::str::Product price with currency'
          ]
        }

        out = Gliner[complex_schema][text, threshold: 0.4]
        product = out.fetch('product').fetch(0)

        expect(entity_text(product.fetch('name'))).to include('iPhone')
        expect(entity_text(product.fetch('storage'))).to include('256')
        expect(entity_text(product.fetch('processor'))).to include('A17')
        expect(entity_text(product.fetch('price'))).to include('1199')
      end

      it 'supports choices and multiple instances' do
        model_dir = ensure_model_dir!
        Gliner.load(model_dir, file: MODEL_FILE)

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

        out = Gliner[schema][text, threshold: 0.2]
        transactions = out.fetch('transaction')
        order = out.fetch('order').fetch(0)

        expect(transactions.length).to be >= 1
        dates = transactions.map { |t| entity_text(t['date']) }.compact
        expect(dates).not_to be_empty
        expect(dates.join(' ')).to include('Jan')

        categories = transactions.map { |t| entity_text(t['category']) }.compact
        categories.each do |category|
          expect(%w[food transport shopping]).to include(category)
        end

        expect(%w[pending processing shipped]).to include(entity_text(order.fetch('status')))
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
    download(hf_resolve_url(MODEL_FILE).to_s, File.join(dir, MODEL_FILE))

    dir
  end

  def hf_resolve_url(filename)
    "https://huggingface.co/#{REPO_ID}/resolve/main/#{REPO_SUBDIR}/#{filename}"
  end

  def download(url, dest)
    return if File.exist?(dest) && File.size?(dest)

    response = HTTPX.plugin(:follow_redirects).with(max_redirects: 5).get(url)
    raise "Download failed: #{url} (status: #{response.status})" unless response.status.between?(200, 299)

    File.binwrite(dest, response.body.to_s)
  end

  def entity_text(value)
    return value.text if value.is_a?(Gliner::Entity)

    value
  end

  def entity_texts(value)
    return [] if value.nil?
    return value.map { |item| entity_text(item) } if value.is_a?(Array)

    [entity_text(value)]
  end
end
