# frozen_string_literal: true

require "test_helper"
require "fileutils"

class Gliner2IntegrationTest < Minitest::Test
  REPO_ID = "0riginalGandalf/gliner2-multi-v1-int8"

  def test_real_model_inference_entities
    skip "set GLINER2_INTEGRATION=1 to run" unless ENV["GLINER2_INTEGRATION"] == "1"

    model_dir = ensure_model_dir!
    model = Gliner2::Model.from_dir(model_dir)

    text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
    labels = ["company", "person", "product", "location"]

    out = model.extract_entities(text, labels, threshold: 0.5)
    entities = out.fetch("entities")

    assert_includes entities.fetch("company"), "Apple"
    assert_includes entities.fetch("person"), "Tim Cook"
    assert_includes entities.fetch("product"), "iPhone 15"
    assert_includes entities.fetch("location"), "Cupertino"
  end

  def test_real_model_inference_classification
    skip "set GLINER2_INTEGRATION=1 to run" unless ENV["GLINER2_INTEGRATION"] == "1"

    model_dir = ensure_model_dir!
    model = Gliner2::Model.from_dir(model_dir)

    out = model.classify_text(
      "This laptop has amazing performance but terrible battery life!",
      { "sentiment" => %w[positive negative neutral] }
    )
    assert_equal "negative", out.fetch("sentiment")
  end

  def test_real_model_inference_structured_extraction
    skip "set GLINER2_INTEGRATION=1 to run" unless ENV["GLINER2_INTEGRATION"] == "1"

    model_dir = ensure_model_dir!
    model = Gliner2::Model.from_dir(model_dir)

    text = "iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199."
    out = model.extract_json(
      text,
      {
        "product" => [
          "name::str::Full product name and model",
          "storage::str::Storage capacity",
          "processor::str::Chip or processor information",
          "price::str::Product price with currency"
        ]
      },
      threshold: 0.4
    )

    product = out.fetch("product").fetch(0)
    assert_includes product.fetch("name"), "iPhone"
    assert_includes product.fetch("storage"), "256"
    assert_includes product.fetch("processor"), "A17"
    assert_includes product.fetch("price"), "$"
  end

  private

  def ensure_model_dir!
    from_env = ENV["GLINER2_MODEL_DIR"]
    return from_env if from_env && !from_env.empty?

    dir = File.expand_path("../tmp/#{REPO_ID.tr('/', '__')}", __dir__)
    FileUtils.mkdir_p(dir)

    download("#{hf_resolve_url("tokenizer.json")}", File.join(dir, "tokenizer.json"))
    download("#{hf_resolve_url("config.json")}", File.join(dir, "config.json"))
    download("#{hf_resolve_url("model_int8.onnx")}", File.join(dir, "model_int8.onnx"))

    dir
  end

  def hf_resolve_url(filename)
    "https://huggingface.co/#{REPO_ID}/resolve/main/#{filename}"
  end

  def download(url, dest)
    return if File.exist?(dest) && File.size?(dest)

    # Use curl to avoid reimplementing HTTP downloads and to support resumes.
    ok = system(
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
      url
    )
    raise "Download failed: #{url}" unless ok
  end
end
