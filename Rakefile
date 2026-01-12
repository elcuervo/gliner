# frozen_string_literal: true

require "rake/testtask"
require "fileutils"

Rake::TestTask.new do |t|
  t.libs << "test"
  t.warning = true
  t.test_files = FileList["test/**/*_test.rb"]
end

task default: :test

namespace :model do
  DEFAULT_REPO_ID = "0riginalGandalf/gliner2-multi-v1-int8"
  DEFAULT_MODEL_FILE = "model_int8.onnx"

  desc "Downloads a test model to tmp/ (REPO_ID=... MODEL_FILE=model_int8.onnx)"
  task :pull do
    repo_id = ENV["REPO_ID"] || DEFAULT_REPO_ID
    model_file = ENV["MODEL_FILE"] || DEFAULT_MODEL_FILE

    dir = File.expand_path("tmp/models/#{repo_id.tr('/', '__')}", __dir__)
    FileUtils.mkdir_p(dir)

    base = "https://huggingface.co/#{repo_id}/resolve/main"
    files = ["tokenizer.json", "config.json", model_file]

    files.each do |file|
      dest = File.join(dir, file)
      next if File.exist?(dest) && File.size?(dest)

      sh(
        "curl",
        "--fail",
        "--location",
        "--retry",
        "3",
        "--retry-delay",
        "1",
        "--continue-at",
        "-",
        "--output",
        dest,
        "#{base}/#{file}"
      )
    end

    puts "Downloaded model to: #{dir}"
    puts "Use with: GLINER_MODEL_DIR=#{dir}"
  end
end

namespace :test do
  desc "Runs real-model integration test (downloads ~357MB unless GLINER_MODEL_DIR is set)"
  task :integration do
    Rake::Task["model:pull"].invoke unless ENV["GLINER_MODEL_DIR"] && !ENV["GLINER_MODEL_DIR"].empty?

    env = { "GLINER_INTEGRATION" => "1" }
    sh env, "ruby", "-Ilib", "-Itest", "test/integration_test.rb"
  end
end

desc "Starts an IRB console (optionally pass MODEL_DIR=/path)"
task :console do
  model_dir = ENV["MODEL_DIR"] || ENV["GLINER_MODEL_DIR"]
  args = ["ruby", "-Ilib", "bin/console"]
  args << model_dir if model_dir && !model_dir.empty?
  sh(*args)
end
