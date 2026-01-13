# gliner (Ruby)

Minimal Ruby inference wrapper for the **GLiNER2** ONNX model using:

- `tokenizers` (https://github.com/ankane/tokenizers-ruby)
- `onnxruntime` (https://github.com/ankane/onnxruntime-ruby)

This gem does **not** ship model weights. Download an ONNX export + tokenizer files separately, then load from a local directory.

## Install

```ruby
gem "gliner"
```

## Usage (entities)

```ruby
require "gliner"

model = Gliner::Model.from_dir("path/to/gliner2-multi-v1-int8")

text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
labels = ["company", "person", "product", "location"]

pp model.extract_entities(text, labels)
```

Expected shape:

```ruby
{"entities"=>{"company"=>["Apple"], "person"=>["Tim Cook"], "product"=>["iPhone 15"], "location"=>["Cupertino"]}}
```

## Usage (classification)

```ruby
result = model.classify_text(
  "This laptop has amazing performance but terrible battery life!",
  { "sentiment" => %w[positive negative neutral] }
)

pp result
```

Expected shape:

```ruby
{"sentiment"=>"negative"}
```

## Usage (structured extraction)

```ruby
text = "iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199."

result = model.extract_json(
  text,
  {
    "product" => [
      "name::str::Full product name and model",
      "storage::str::Storage capacity",
      "processor::str::Chip or processor information",
      "price::str::Product price with currency"
    ]
  }
)

pp result
```

Expected shape:

```ruby
{"product"=>[{"name"=>"iPhone 15 Pro Max", "storage"=>"256GB", "processor"=>"A17 Pro chip", "price"=>"$1199"}]}
```

## Model files

This implementation expects a directory containing:

- `tokenizer.json`
- `model.onnx` or `model_int8.onnx`
- (optional) `config.json` with `max_width` and `max_seq_len`

One publicly available ONNX export is `cuerbot/gliner2-multi-v1-int8` on Hugging Face.

## Integration test (real model)

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

## Console (REPL)

Start an IRB session with the gem loaded:

```bash
rake console MODEL_DIR=/path/to/model_dir
```

If you omit `MODEL_DIR`, the console auto-downloads a public test model (configurable):

```bash
rake console
# or:
GLINER_REPO_ID=cuerbot/gliner2-multi-v1-int8 GLINER_MODEL_FILE=model_int8.onnx rake console
```

Or:

```bash
ruby -Ilib bin/console /path/to/model_dir
```
