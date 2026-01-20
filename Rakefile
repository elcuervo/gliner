# frozen_string_literal: true

require 'rspec/core/rake_task'
require 'fileutils'
require 'httpx'

RSpec::Core::RakeTask.new(:spec)

task default: :spec

namespace :model do
  DEFAULT_REPO_ID = 'cuerbot/gliner2-multi-v1'
  DEFAULT_MODEL_FILE = 'model_fp16.onnx'
  DEFAULT_MODEL_SUBDIR = 'onnx'

  desc 'Downloads a test model to tmp/ (REPO_ID=... MODEL_FILE=model_fp16.onnx)'
  task :pull do
    repo_id = ENV['REPO_ID'] || DEFAULT_REPO_ID
    model_file = ENV['MODEL_FILE'] || DEFAULT_MODEL_FILE
    model_subdir = ENV['MODEL_SUBDIR'] || DEFAULT_MODEL_SUBDIR

    dir = File.expand_path("tmp/models/#{repo_id.tr('/', '__')}", __dir__)
    FileUtils.mkdir_p(dir)

    base = "https://huggingface.co/#{repo_id}/resolve/main"
    base = "#{base}/#{model_subdir}" unless model_subdir.nil? || model_subdir.empty?
    files = ['tokenizer.json', 'config.json', model_file]
    client = HTTPX.plugin(:follow_redirects).with(max_redirects: 5)

    files.each do |file|
      dest = File.join(dir, file)
      next if File.exist?(dest) && File.size?(dest)

      response = client.get("#{base}/#{file}")
      raise "Download failed: #{base}/#{file} (status: #{response.status})" unless response.status.between?(200, 299)

      File.binwrite(dest, response.body.to_s)
    end

    puts "Downloaded model to: #{dir}"
    puts "Use with: GLINER_MODEL_DIR=#{dir}"
  end
end

namespace :spec do
  desc 'Runs real-model integration test (downloads ~357MB unless GLINER_MODEL_DIR is set)'
  task :integration do
    repo_id = ENV['REPO_ID'] || DEFAULT_REPO_ID
    model_file = ENV['MODEL_FILE'] || DEFAULT_MODEL_FILE
    model_subdir = ENV['MODEL_SUBDIR'] || DEFAULT_MODEL_SUBDIR

    Rake::Task['model:pull'].invoke unless ENV['GLINER_MODEL_DIR'] && !ENV['GLINER_MODEL_DIR'].empty?

    model_dir = ENV['GLINER_MODEL_DIR']
    model_dir = File.expand_path("tmp/models/#{repo_id.tr('/', '__')}", __dir__) if model_dir.nil? || model_dir.empty?

    env = {
      'GLINER_INTEGRATION' => '1',
      'GLINER_MODEL_DIR' => model_dir,
      'GLINER_MODEL_FILE' => model_file,
      'GLINER_MODEL_SUBDIR' => model_subdir
    }
    sh env, 'rspec', 'spec/integration_spec.rb'
  end
end

desc 'Starts an IRB console (optionally pass MODEL_DIR=/path)'
task :console do
  model_dir = ENV['MODEL_DIR'] || ENV.fetch('GLINER_MODEL_DIR', nil)
  args = ['ruby', '-Ilib', 'bin/console']
  args << model_dir if model_dir && !model_dir.empty?
  sh(*args)
end

namespace :gem do
  desc 'Build the gem'
  task :build do
    sh 'gem', 'build', 'gliner.gemspec'
  end

  desc 'Build and push the gem to RubyGems'
  task push: :build do
    require_relative 'lib/gliner/version'
    gem_file = "gliner-#{Gliner::VERSION}.gem"
    sh 'gem', 'push', gem_file
  end
end
