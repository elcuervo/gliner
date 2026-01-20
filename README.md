# GLiNER
[![tests](https://github.com/elcuervo/gliner/actions/workflows/tests.yml/badge.svg)](https://github.com/elcuervo/gliner/actions/workflows/tests.yml)
![Gem Version](https://img.shields.io/gem/v/gliner)

![](https://images.unsplash.com/photo-1625768376503-68d2495d78c5?q=80&w=2225&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D)

## Install

```ruby
gem "gliner"
```

## Usage

### Entities

```ruby
require "gliner"

Gliner.configure do |config|
  config.threshold = 0.2
  # By default, the gem downloads the default model to .cache/
  # Or set a local path explicitly:
  # config.model = "/path/to/gliner2-multi-v1"
  config.variant = :fp16
end

text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
labels = ["company", "person", "product", "location"]

model = Gliner[labels]
entities = model[text]

pp entities["person"]
# => [#<data Gliner::Entity ...>]

entities["person"].first.text
# => "Tim Cook"

entities["person"].first.probability
# => 92.4

entities["person"].first.offsets
# => [10, 18]
```

You can also pass per-entity configs:

```ruby
labels = {
  email: { description: "Email addresses", dtype: "list", threshold: 0.9 },
  person: { description: "Person names", dtype: "str" }
}

model = Gliner[labels]
entities = model["Email John Doe at john@example.com.", threshold: 0.5]

entities["person"].text
# => "John Doe"

entities["email"].map(&:text)
# => ["john@example.com"]
```

### Classification

```ruby
model = Gliner.classify[
  { sentiment: %w[positive negative neutral] }
]

result = model["This laptop has amazing performance but terrible battery life!"]

pp result

# => { sentiment: #<data Gliner::Label ...> }

result["sentiment"].label
# => "negative"

result["sentiment"].probability
# => 87.1
```

Multiple classification tasks:

```ruby
text = "Breaking: Tech giant announces major layoffs amid market downturn"

tasks = {
  sentiment: %w[positive negative neutral],
  urgency: %w[high medium low],
  category: { labels: %w[tech finance politics sports], multi_label: false }
}

results = Gliner.classify[tasks][text]

results.transform_values { |value| value.label }
# => { sentiment: "negative", urgency: "high", category: "tech" }
```

### Structured extraction

```ruby
text = "iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199."

structure = {
  product: [
    "name::str::Full product name and model",
    "storage::str::Storage capacity",
    "processor::str::Chip or processor information",
    "price::str::Product price with currency"
  ]
}

result = Gliner[structure][text]
product = result.fetch("product").first

pp result

product["name"].text
# => "iPhone 15 Pro Max"

product["storage"].text
# => "256GB"

product["processor"].text
# => "A17 Pro"

product["price"].text
# => "1199"
```

Choices can be included in field specs:

```ruby
result = Gliner[{ order: ["status::[pending|processing|shipped]::str"] }]["Status: shipped"]

result.fetch("order").first["status"].text
# shipped
```

## Model files

This implementation expects a directory containing:

- `tokenizer.json`
- `model.onnx`, `model_fp16.onnx`, or `model_int8.onnx`
- (optional) `config.json` with `max_width` and `max_seq_len`

One publicly available ONNX export is `cuerbot/gliner2-multi-v1` on Hugging Face.
By default, `model_fp16.onnx` is used; set `config.variant` (or `GLINER_MODEL_FILE`) to override.
Variants map to files as: `:fp16` → `model_fp16.onnx`, `:fp32` → `model.onnx`, `:int8` → `model_int8.onnx`.

You can also configure the model source directly:

```ruby
Gliner.configure do |config|
  config.model = "/path/to/model_dir"
  config.variant = :int8
end
```

## Integration test

Downloads a public ONNX export and runs a real inference:

```bash
rake test:integration
```

To download the model separately (for console testing, etc):

```bash
rake model:pull
```

To reuse an existing local download:

```bash
GLINER_MODEL_DIR=/path/to/model_dir rake test:integration
```

## Console

Start an IRB session with the gem loaded:

```bash
rake console MODEL_DIR=/path/to/model_dir
```

If you omit `MODEL_DIR`, the console auto-downloads a public test model (configurable):

```bash
rake console
# or:
GLINER_REPO_ID=cuerbot/gliner2-multi-v1 GLINER_MODEL_FILE=model_fp16.onnx rake console
```

Or:

```bash
ruby -Ilib bin/console /path/to/model_dir
```
