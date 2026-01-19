# frozen_string_literal: true

require_relative 'lib/gliner/version'

Gem::Specification.new do |spec|
  spec.name = 'gliner'
  spec.version = Gliner::VERSION
  spec.authors = ['elcuervo']

  spec.summary = 'Schema-based information extraction (GLiNER2) via ONNX Runtime'
  spec.description = 'Basic Ruby inference wrapper for the GLiNER2 ONNX model.'
  spec.homepage = 'https://github.com/elcuervo/gliner'
  spec.license = 'MIT'
  spec.required_ruby_version = '>= 3.2'

  spec.files = Dir.glob('lib/**/*') + Dir.glob('bin/*') + %w[README.md LICENSE gliner.gemspec]
  spec.require_paths = ['lib']

  spec.add_dependency 'onnxruntime', '~> 0.10'
  spec.add_dependency 'tokenizers', '~> 0.6'

  spec.add_development_dependency 'httpx', '~> 1.0'
  spec.add_development_dependency 'rake', '~> 13.0'
  spec.add_development_dependency 'rspec', '~> 3.13'
  spec.add_development_dependency 'rubocop', '~> 1.50'
end
