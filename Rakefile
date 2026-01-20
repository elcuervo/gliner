# frozen_string_literal: true

require 'rspec/core/rake_task'
require 'fileutils'
require 'httpx'

RSpec::Core::RakeTask.new(:spec)

task default: :spec

namespace :spec do
  desc 'Runs real-model integration test (downloads ~357MB unless GLINER_MODEL_DIR is set)'
  task :integration do
    env = { 'GLINER_INTEGRATION' => '1', }

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
