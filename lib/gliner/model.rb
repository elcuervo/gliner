# frozen_string_literal: true

require "json"
require "onnxruntime"
require "tokenizers"

module Gliner
  class Model
    SPECIAL_TOKENS = [
      "[SEP_STRUCT]",
      "[SEP_TEXT]",
      "[P]",
      "[C]",
      "[E]",
      "[R]",
      "[L]",
      "[EXAMPLE]",
      "[OUTPUT]",
      "[DESCRIPTION]"
    ].freeze

    DEFAULT_MAX_WIDTH = 8
    DEFAULT_MAX_SEQ_LEN = 512

    TASK_TYPE_ENTITIES = 0
    TASK_TYPE_CLASSIFICATION = 1
    TASK_TYPE_JSON = 2

    def self.from_dir(dir, model_filename: nil)
      tokenizer_path = File.join(dir, "tokenizer.json")
      raise Error, "Missing tokenizer.json in #{dir}" unless File.exist?(tokenizer_path)

      config_path = File.join(dir, "config.json")
      config = File.exist?(config_path) ? JSON.parse(File.read(config_path)) : {}

      model_path =
        if model_filename
          File.join(dir, model_filename)
        elsif File.exist?(File.join(dir, "model_int8.onnx"))
          File.join(dir, "model_int8.onnx")
        else
          File.join(dir, "model.onnx")
        end
      raise Error, "Missing model file in #{dir} (expected model.onnx or model_int8.onnx)" unless File.exist?(model_path)

      new(
        model_path: model_path,
        tokenizer_path: tokenizer_path,
        max_width: config.fetch("max_width", DEFAULT_MAX_WIDTH),
        max_seq_len: config.fetch("max_seq_len", DEFAULT_MAX_SEQ_LEN)
      )
    end

    def initialize(model_path:, tokenizer_path:, max_width: DEFAULT_MAX_WIDTH, max_seq_len: DEFAULT_MAX_SEQ_LEN)
      @model_path = model_path
      @tokenizer = Tokenizers.from_file(tokenizer_path)
      @word_pre_tokenizer = Tokenizers::PreTokenizers::BertPreTokenizer.new

      missing_specials = SPECIAL_TOKENS.reject { |t| @tokenizer.token_to_id(t) }
      unless missing_specials.empty?
        raise Error, "Tokenizer missing special tokens: #{missing_specials.join(', ')}"
      end

      @max_width = Integer(max_width)
      @max_seq_len = Integer(max_seq_len)

      @session = OnnxRuntime::InferenceSession.new(@model_path)
      validate_io!
    end

    # Supports:
    # - Array: ["company", "person"]
    # - Hash: {"company"=>"desc", "person"=>"desc"}
    #
    # Returns: {"entities" => {"label" => ["span", ...], ...}}
    def extract_entities(text, entity_types, threshold: 0.5, format_results: true, include_confidence: false, include_spans: false)
      labels, label_descriptions = normalize_labels_with_descriptions(entity_types)
      prompt = build_prompt("entities", label_descriptions)

      schema_tokens = schema_tokens_for(prompt: prompt, labels: labels, label_prefix: "[E]")
      prepared = prepare_inputs(text, schema_tokens)

      logits = run_onnx(
        input_ids: prepared[:input_ids],
        attention_mask: prepared[:attention_mask],
        words_mask: prepared[:words_mask],
        text_lengths: [prepared[:text_len]],
        task_type: TASK_TYPE_ENTITIES,
        label_positions: label_positions_for(prepared[:word_ids], labels.length),
        label_mask: Array.new(labels.length, 1)
      )

      entities = extract_span_values(
        logits: logits,
        labels: labels,
        pos_to_word_index: prepared[:pos_to_word_index],
        start_map: prepared[:start_map],
        end_map: prepared[:end_map],
        original_text: prepared[:original_text],
        text_len: prepared[:text_len],
        threshold: threshold,
        include_confidence: include_confidence,
        include_spans: include_spans,
        dtype: :list
      )

      { "entities" => entities }
    end

    # Supports:
    # - {"sentiment" => ["positive","negative","neutral"]}
    # - {"aspects" => {"labels"=>[...], "multi_label"=>true, "cls_threshold"=>0.4}}
    # - {"sentiment" => {"labels"=>{"positive"=>"desc", ...}}}
    #
    # Returns:
    # - Single-label: {"sentiment"=>"negative"}
    # - Multi-label: {"aspects"=>["camera","price"]}
    def classify_text(text, tasks, threshold: 0.5, format_results: true, include_confidence: false, include_spans: false)
      raise Error, "tasks must be a Hash" unless tasks.is_a?(Hash)

      out = {}
      tasks.each do |task_name, config|
        parsed = parse_classification_task(task_name, config)
        schema_tokens = schema_tokens_for(prompt: parsed[:prompt], labels: parsed[:labels], label_prefix: "[L]")
        prepared = prepare_inputs(text, schema_tokens)

        logits = run_onnx(
          input_ids: prepared[:input_ids],
          attention_mask: prepared[:attention_mask],
          words_mask: prepared[:words_mask],
          text_lengths: [prepared[:text_len]],
          task_type: TASK_TYPE_CLASSIFICATION,
          label_positions: label_positions_for(prepared[:word_ids], parsed[:labels].length),
          label_mask: Array.new(parsed[:labels].length, 1)
        )

        scores = classification_scores(
          logits: logits,
          labels: parsed[:labels],
          pos_to_word_index: prepared[:pos_to_word_index],
          text_len: prepared[:text_len],
          threshold: parsed[:cls_threshold]
        )

        out[task_name.to_s] = format_classification(
          scores,
          labels: parsed[:labels],
          multi_label: parsed[:multi_label],
          include_confidence: include_confidence,
          cls_threshold: parsed[:cls_threshold]
        )
      end

      out
    end

    # Supports (Python-style):
    #
    #   model.extract_json(text, {
    #     "product" => [
    #       "name::str::Full product name and model",
    #       "storage::str::Storage capacity",
    #       "processor::str::Chip or processor information",
    #       "price::str::Product price with currency"
    #     ]
    #   })
    #
    # Returns:
    #   {"product"=>[{"name"=>"...", ...}]}
    def extract_json(text, structures, threshold: 0.5, format_results: true, include_confidence: false, include_spans: false)
      raise Error, "structures must be a Hash" unless structures.is_a?(Hash)

      normalized_text = normalize_text(text)
      out = {}

      structures.each do |parent, fields|
        parent_name = parent.to_s
        parsed_fields = Array(fields).map { |spec| parse_field_spec(spec.to_s) }
        labels = parsed_fields.map { |f| f[:name] }
        descs = parsed_fields.filter_map { |f| f[:description] ? [f[:name], f[:description]] : nil }.to_h

        prompt = build_prompt(parent_name, descs)
        schema_tokens = schema_tokens_for(prompt: prompt, labels: labels, label_prefix: "[C]")
        prepared = prepare_inputs(normalized_text, schema_tokens, already_normalized: true)

        logits = run_onnx(
          input_ids: prepared[:input_ids],
          attention_mask: prepared[:attention_mask],
          words_mask: prepared[:words_mask],
          text_lengths: [prepared[:text_len]],
          task_type: TASK_TYPE_JSON,
          label_positions: label_positions_for(prepared[:word_ids], labels.length),
          label_mask: Array.new(labels.length, 1)
        )

        spans_by_label = extract_spans_by_label(
          logits: logits,
          labels: labels,
          pos_to_word_index: prepared[:pos_to_word_index],
          start_map: prepared[:start_map],
          end_map: prepared[:end_map],
          original_text: prepared[:original_text],
          text_len: prepared[:text_len],
          threshold: threshold
        )

        obj = {}
        parsed_fields.each do |field|
          key = field[:name]
          spans = spans_by_label.fetch(key)

          if field[:dtype] == :str
            best = choose_best_span(spans)
            obj[key] = format_single_span(best, include_confidence: include_confidence, include_spans: include_spans)
          else
            obj[key] = format_spans(spans, include_confidence: include_confidence, include_spans: include_spans)
          end
        end

        out[parent_name] = [obj]
      end

      out
    end

    private

    def validate_io!
      input_names = @session.inputs.map { |i| i[:name] }
      expected_inputs = %w[input_ids attention_mask words_mask text_lengths task_type label_positions label_mask]
      missing = expected_inputs - input_names
      raise Error, "Model missing inputs: #{missing.join(', ')}" unless missing.empty?

      output_names = @session.outputs.map { |o| o[:name] }
      raise Error, "Model missing output: logits" unless output_names.include?("logits")
    end

    def normalize_text(text)
      str = text.to_s
      str = "." if str.empty?
      str.end_with?(".", "!", "?") ? str : "#{str}."
    end

    def normalize_labels_with_descriptions(labels)
      case labels
      when Array
        [labels.map(&:to_s), {}]
      when String, Symbol
        [[labels.to_s], {}]
      when Hash
        names = labels.keys.map(&:to_s)
        descs = labels.transform_keys(&:to_s).transform_values { |v| v.is_a?(String) ? v : nil }.compact
        [names, descs]
      else
        raise Error, "labels must be a String, Array, or Hash"
      end
    end

    def split_words(text)
      text = text.to_s
      tokens = []
      starts = []
      ends = []

      @word_pre_tokenizer.pre_tokenize_str(text).each do |(token, (start_pos, end_pos))|
        token = token.to_s.downcase
        next if token.empty?
        tokens << token
        starts << start_pos
        ends << end_pos
      end

      [tokens, starts, ends]
    end

    def build_prompt(base, label_descriptions)
      prompt = base.to_s
      label_descriptions.to_h.each do |label, desc|
        next if desc.to_s.empty?
        prompt += " [DESCRIPTION] #{label}: #{desc}"
      end
      prompt
    end

    def schema_tokens_for(prompt:, labels:, label_prefix:)
      tokens = ["(", "[P]", prompt.to_s, "("]
      labels.each do |label|
        tokens << label_prefix
        tokens << label.to_s
      end
      tokens.concat([")", ")"])
      tokens
    end

    def encode_pretokenized(tokens)
      enc = @tokenizer.encode(tokens, is_pretokenized: true, add_special_tokens: false)
      { ids: enc.ids, word_ids: enc.word_ids }
    end

    def truncate_inputs!(input_ids, word_ids, max_len:)
      return { input_ids: input_ids, word_ids: word_ids } if input_ids.length <= max_len
      { input_ids: input_ids.take(max_len), word_ids: word_ids.take(max_len) }
    end

    def prepare_inputs(text, schema_tokens, already_normalized: false)
      normalized_text = already_normalized ? text.to_s : normalize_text(text)
      words, start_map, end_map = split_words(normalized_text)
      combined_tokens = schema_tokens + ["[SEP_TEXT]"] + words

      encoded = encode_pretokenized(combined_tokens)
      input_ids = encoded[:ids]
      word_ids = encoded[:word_ids]

      truncated = truncate_inputs!(input_ids, word_ids, max_len: @max_seq_len)
      input_ids = truncated[:input_ids]
      word_ids = truncated[:word_ids]

      text_start_combined = schema_tokens.length + 1
      full_text_len = words.length
      effective_text_len = infer_effective_text_len(word_ids, text_start_combined, full_text_len)

      {
        input_ids: input_ids,
        word_ids: word_ids,
        attention_mask: Array.new(input_ids.length, 1),
        words_mask: build_words_mask(word_ids, text_start_combined),
        pos_to_word_index: build_pos_to_word_index(word_ids, text_start_combined),
        start_map: start_map,
        end_map: end_map,
        original_text: normalized_text,
        text_len: effective_text_len
      }
    end

    def build_words_mask(word_ids, text_start_combined)
      mask = Array.new(word_ids.length, 0)
      last_wid = nil
      word_ids.each_with_index do |wid, i|
        next if wid.nil?
        if wid != last_wid
          mask[i] = 1 if wid >= text_start_combined
          last_wid = wid
        end
      end
      mask
    end

    def build_pos_to_word_index(word_ids, text_start_combined)
      map = Array.new(word_ids.length)
      seen = {}
      word_ids.each_with_index do |wid, i|
        next if wid.nil?
        next if seen.key?(wid)
        seen[wid] = true
        map[i] = wid - text_start_combined if wid >= text_start_combined
      end
      map
    end

    def infer_effective_text_len(word_ids, text_start_combined, full_text_len)
      max_text_wid = word_ids.compact.select { |wid| wid >= text_start_combined }.max
      return full_text_len if max_text_wid.nil?

      present = (max_text_wid - text_start_combined) + 1
      [present, full_text_len].min
    end

    def run_onnx(input_ids:, attention_mask:, words_mask:, text_lengths:, task_type:, label_positions:, label_mask:)
      inputs = {
        input_ids: [input_ids],
        attention_mask: [attention_mask],
        words_mask: [words_mask],
        text_lengths: [text_lengths],
        task_type: [task_type],
        label_positions: [label_positions],
        label_mask: [label_mask]
      }

      # onnxruntime-ruby returns outputs in the same order as output_names
      out = @session.run(["logits"], inputs)
      out.fetch(0)
    end

    def sigmoid(x) = 1.0 / (1.0 + Math.exp(-x))

    # Returns spans as [text, score, char_start, char_end]
    def find_spans_for_label(logits:, label_index:, pos_to_word_index:, start_map:, end_map:, original_text:, text_len:, threshold:)
      spans = []

      seq_len = logits[0].length
      (0...seq_len).each do |pos|
        start_word = pos_to_word_index[pos]
        next if start_word.nil?

        (0...@max_width).each do |width|
          end_word = start_word + width
          next if end_word >= text_len

          score = sigmoid(logits[0][pos][width][label_index])
          next if score < threshold

          char_start = start_map[start_word]
          char_end = end_map[end_word]
          next if char_start.nil? || char_end.nil?

          text_span = original_text[char_start...char_end].to_s.strip
          next if text_span.empty?

          spans << [text_span, score, char_start, char_end]
        end
      end

      spans
    end

    def label_positions_for(word_ids, label_count)
      label_count.times.map do |i|
        combined_idx = 4 + (i * 2)
        pos = word_ids.index(combined_idx)
        raise Error, "Could not locate label position at combined index #{combined_idx}" if pos.nil?
        pos
      end
    end

    def extract_spans_by_label(logits:, labels:, pos_to_word_index:, start_map:, end_map:, original_text:, text_len:, threshold:)
      out = {}
      labels.each_with_index do |label, label_index|
        out[label.to_s] = find_spans_for_label(
          logits: logits,
          label_index: label_index,
          pos_to_word_index: pos_to_word_index,
          start_map: start_map,
          end_map: end_map,
          original_text: original_text,
          text_len: text_len,
          threshold: threshold
        )
      end
      out
    end

    def extract_span_values(logits:, labels:, pos_to_word_index:, start_map:, end_map:, original_text:, text_len:, threshold:,
                            include_confidence:, include_spans:, dtype:)
      spans_by_label = extract_spans_by_label(
        logits: logits,
        labels: labels,
        pos_to_word_index: pos_to_word_index,
        start_map: start_map,
        end_map: end_map,
        original_text: original_text,
        text_len: text_len,
        threshold: threshold
      )

      spans_by_label.transform_values do |spans|
        if dtype == :str
          format_single_span(choose_best_span(spans), include_confidence: include_confidence, include_spans: include_spans)
        else
          format_spans(spans, include_confidence: include_confidence, include_spans: include_spans)
        end
      end
    end

    def choose_best_span(spans)
      return nil if spans.empty?
      sorted = spans.sort_by { |(t, score, start_pos, end_pos)| [-score, (end_pos - start_pos), t.length] }
      best = sorted[0]
      best_score = best[1]
      near = sorted.take_while { |s| (best_score - s[1]) <= 0.02 }
      near.min_by { |(t, score, start_pos, end_pos)| [(end_pos - start_pos), -score, t.length] } || best
    end

    def format_single_span(span, include_confidence:, include_spans:)
      return nil if span.nil?
      text, score, start_pos, end_pos = span

      if include_spans && include_confidence
        { "text" => text, "confidence" => score, "start" => start_pos, "end" => end_pos }
      elsif include_spans
        { "text" => text, "start" => start_pos, "end" => end_pos }
      elsif include_confidence
        { "text" => text, "confidence" => score }
      else
        text
      end
    end

    def parse_field_spec(spec)
      parts = spec.split("::")
      name = parts[0].to_s
      dtype = :list
      description = nil
      dtype_explicit = false

      parts.drop(1).each do |part|
        part = part.to_s
        if part == "str"
          dtype = :str
          dtype_explicit = true
        elsif part == "list"
          dtype = :list
          dtype_explicit = true
        elsif part.start_with?("[") && part.end_with?("]")
          # choices are currently ignored, but keep dtype default compatible with Python parser
          dtype = :str unless dtype_explicit
        elsif description.nil?
          description = part
        else
          description += "::#{part}"
        end
      end

      { name: name, dtype: dtype, description: description }
    end

    def parse_classification_task(task_name, config)
      multi_label = false
      cls_threshold = 0.5
      labels = []
      label_descs = {}

      case config
      when Array
        labels = config.map(&:to_s)
      when Hash
        cfg = config.transform_keys(&:to_s)
        if cfg.key?("labels")
          multi_label = !!cfg["multi_label"]
          cls_threshold = cfg["cls_threshold"] ? Float(cfg["cls_threshold"]) : cls_threshold

          raw_labels = cfg["labels"]
          if raw_labels.is_a?(Array)
            labels = raw_labels.map(&:to_s)
          elsif raw_labels.is_a?(Hash)
            labels = raw_labels.keys.map(&:to_s)
            label_descs = raw_labels.transform_keys(&:to_s).transform_values(&:to_s)
          else
            raise Error, "classification task #{task_name.inspect} must include labels"
          end
        else
          # Support {"sentiment"=>{"positive"=>"...", "negative"=>"..."}}
          labels = config.keys.map(&:to_s)
          label_descs = config.transform_keys(&:to_s).transform_values(&:to_s)
        end
      else
        raise Error, "classification task #{task_name.inspect} must be an Array or Hash"
      end

      {
        labels: labels,
        multi_label: multi_label,
        cls_threshold: cls_threshold,
        prompt: build_prompt(task_name.to_s, label_descs)
      }
    end

    def classification_scores(logits:, labels:, pos_to_word_index:, text_len:, threshold:)
      scores = []

      labels.each_index do |label_index|
        max = -Float::INFINITY
        seq_len = logits[0].length
        (0...seq_len).each do |pos|
          start_word = pos_to_word_index[pos]
          next if start_word.nil?

          (0...@max_width).each do |width|
            end_word = start_word + width
            next if end_word >= text_len
            s = sigmoid(logits[0][pos][width][label_index])
            max = s if s > max
          end
        end
        scores << max
      end

      scores
    end

    def format_classification(scores, labels:, multi_label:, include_confidence:, cls_threshold:)
      pairs = scores.each_with_index.map { |s, i| [labels.fetch(i), s] }
      pairs.sort_by! { |(_i, s)| -s }

      if multi_label
        chosen = pairs.select { |(_label, s)| s >= cls_threshold }
        chosen = [pairs.first] if chosen.empty? && pairs.first
        chosen.map! { |(label, s)| include_confidence ? { "label" => label, "confidence" => s } : label }
      else
        label, s = pairs.first
        include_confidence ? { "label" => label, "confidence" => s } : label
      end
    end

    def format_spans(spans, include_confidence:, include_spans:)
      return [] if spans.empty?

      sorted = spans.sort_by { |(_, score, _, _)| -score }
      selected = []

      sorted.each do |text, score, start_pos, end_pos|
        overlaps = selected.any? { |(_, _, s, e)| !(end_pos <= s || start_pos >= e) }
        next if overlaps
        selected << [text, score, start_pos, end_pos]
      end

      if include_spans && include_confidence
        selected.map { |t, s, st, en| { "text" => t, "confidence" => s, "start" => st, "end" => en } }
      elsif include_spans
        selected.map { |t, _s, st, en| { "text" => t, "start" => st, "end" => en } }
      elsif include_confidence
        selected.map { |t, s, _st, _en| { "text" => t, "confidence" => s } }
      else
        selected.map(&:first)
      end
    end
  end
end
