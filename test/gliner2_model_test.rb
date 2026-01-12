# frozen_string_literal: true

require "test_helper"

class Gliner2ModelTest < Minitest::Test
  def test_extract_entities_raises_without_files
    err = assert_raises(Gliner2::Error) { Gliner2::Model.from_dir("does-not-exist") }
    assert_match(/Missing tokenizer\.json/, err.message)
  end

  def test_overlap_filtering
    model = allocate_model_for_unit_test
    spans = [
      ["Tim", 0.8, 0, 3],
      ["Tim Cook", 0.9, 0, 8],
      ["Cook", 0.7, 4, 8]
    ]

    out = model.__send__(:format_spans, spans, include_confidence: false, include_spans: false)
    assert_equal ["Tim Cook"], out
  end

  private

  def allocate_model_for_unit_test
    # Avoid requiring a real ONNX model in unit tests.
    model = Gliner2::Model.allocate
    model.instance_variable_set(:@max_width, 8)
    model
  end
end
