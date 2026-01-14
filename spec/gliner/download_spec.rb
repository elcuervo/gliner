# frozen_string_literal: true

require 'spec_helper'
require 'httpx'
require 'fileutils'
require 'tmpdir'

describe 'Download Helper' do
  # Helper method from integration_spec
  def download(url, dest)
    return if File.exist?(dest) && File.size?(dest)

    temp_dest = "#{dest}.partial"
    resume_from = File.exist?(temp_dest) ? File.size(temp_dest) : 0

    http = HTTPX.plugin(:retries).with(
      max_retries: 3,
      retry_after: 1,
      timeout: { operation_timeout: 300 }
    )

    headers = {}
    headers['Range'] = "bytes=#{resume_from}-" if resume_from > 0

    response = http.get(url, headers: headers)

    raise "Download failed: #{url} (status: #{response.status})" unless response.status.between?(200, 299)

    mode = resume_from > 0 && response.status == 206 ? 'ab' : 'wb'
    File.open(temp_dest, mode) { |f| f.write(response.body.to_s) }

    File.rename(temp_dest, dest)
  rescue StandardError => e
    File.delete(temp_dest) if File.exist?(temp_dest) && resume_from.zero?
    raise "Download failed: #{url} - #{e.message}"
  end

  describe '#download' do
    it 'skips download if file exists and has content' do
      Dir.mktmpdir do |dir|
        dest = File.join(dir, 'existing.txt')
        File.write(dest, 'content')

        # Should not make any HTTP request
        download('https://example.com/file', dest)

        expect(File.read(dest)).to eq('content')
      end
    end

    it 'raises error for invalid URL' do
      Dir.mktmpdir do |dir|
        dest = File.join(dir, 'file.txt')

        expect do
          download('https://invalid-domain-that-does-not-exist-12345.com/file', dest)
        end.to raise_error(/Download failed/)
      end
    end

    it 'creates destination file after successful download', :skip do
      # This test requires network access and a valid URL
      # Skip by default, can be enabled for manual testing
      Dir.mktmpdir do |dir|
        dest = File.join(dir, 'test.txt')
        url = 'https://httpbin.org/robots.txt'

        download(url, dest)

        expect(File.exist?(dest)).to be true
        expect(File.size(dest)).to be > 0
      end
    end

    it 'cleans up partial file on initial download failure' do
      Dir.mktmpdir do |dir|
        dest = File.join(dir, 'file.txt')
        partial = "#{dest}.partial"

        expect do
          download('https://invalid-domain-12345.com/file', dest)
        end.to raise_error(/Download failed/)

        expect(File.exist?(partial)).to be false
      end
    end
  end
end
