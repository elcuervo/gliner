# frozen_string_literal: true

require 'gliner'

RSpec.configure do |config|
  config.example_status_persistence_file_path = '.rspec_status'

  config.expect_with :rspec do |c|
    c.syntax = :expect
  end

  config.before(:suite) do
    Gliner.configure do |gliner_config|
      gliner_config.variant = :fp16
    end
  end
end
