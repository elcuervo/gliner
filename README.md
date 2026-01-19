# Gliner

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
  config.model_dir = "/path/to/gliner2-multi-v1"
  config.model_file = "model.onnx"
end

Gliner.load("path/to/gliner2-multi-v1")

text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
labels = ["company", "person", "product", "location"]

model = Gliner[labels]
pp model[text]
```

Expected shape:

```ruby
{"entities"=>{"company"=>["Apple"], "person"=>["Tim Cook"], "product"=>["iPhone 15"], "location"=>["Cupertino"]}}
```

You can also pass per-entity configs:

```ruby
labels = {
  "email" => { "description" => "Email addresses", "dtype" => "list", "threshold" => 0.9 },
  "person" => { "description" => "Person names", "dtype" => "str" }
}

model = Gliner[labels]
pp model["Email John Doe at john@example.com.", threshold: 0.5]
```

### Classification

```ruby
model = Gliner.classify[
  { "sentiment" => %w[positive negative neutral] }
]

result = model["This laptop has amazing performance but terrible battery life!"]

pp result
```

Expected shape:

```ruby
{"sentiment"=>"negative"}
```

### Structured extraction

```ruby
text = "iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199."

structure = {
  "product" => [
    "name::str::Full product name and model",
    "storage::str::Storage capacity",
    "processor::str::Chip or processor information",
    "price::str::Product price with currency"
  ]
}

result = Gliner[structure][text]

pp result
```

Expected shape:

```ruby
{"product"=>[{"name"=>"iPhone 15 Pro Max", "storage"=>"256GB", "processor"=>"A17 Pro chip", "price"=>"$1199"}]}
```

Choices can be included in field specs:

```ruby
result = Gliner[{ "order" => ["status::[pending|processing|shipped]::str"] }]["Status: shipped"]
```

## Model files

This implementation expects a directory containing:

- `tokenizer.json`
- `model.onnx` or `model_int8.onnx`
- (optional) `config.json` with `max_width` and `max_seq_len`

One publicly available ONNX export is `cuerbot/gliner2-multi-v1` on Hugging Face.
By default, `model_int8.onnx` is used; set `config.model_file` or `GLINER_MODEL_FILE` to override.

To make CI runs more reproducible across Linux hosts, you can force a single-threaded,
sequential ONNX Runtime session (this is also enabled automatically when `CI` is set):

```bash
GLINER_DETERMINISTIC=1
# or:
GLINER_ORT_THREADS=1
```

You can also configure the model directory in code:

```ruby
Gliner.configure do |config|
  config.model_dir = "/path/to/model_dir"
  config.model_file = "model_int8.onnx"
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
GLINER_REPO_ID=cuerbot/gliner2-multi-v1 GLINER_MODEL_FILE=model_int8.onnx rake console
```

Or:

```bash
ruby -Ilib bin/console /path/to/model_dir
```
