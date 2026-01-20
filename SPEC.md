# Single entities

```ruby
text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."

entities = ["company", "person", "product", "location"]
model = Gliner[entities]
entities = model[text]

# => {"company"=>[#<data Gliner::Entity ...>], "person"=>[#<data Gliner::Entity ...>], ...}

entities["person"].first.text
# => "Tim Cook"
```

# Per entity config

```ruby
text = "Email John Doe at john@example.com."

entities = {
  email: { description: "Email addresses", dtype: "list", threshold: 0.9 },
  person: { description: "Person names", dtype: "str" }
}

model = Gliner[entities]
entities = model[text]

entities["person"].text
# => "John Doe"

entities["email"].map(&:text)
# => ["john@example.com"]
```

# Classfication

```ruby
text = "This laptop has amazing performance but terrible battery life!",
concept = { "sentiment" => %w[positive negative neutral] }
model = Gliner.classify[concept]
result = model[text]

# => {"sentiment"=>#<data Gliner::Label ...>}

result["sentiment"].label
# => "negative"

result["sentiment"].confidence
# => 87.1
```

# Structured

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

model = Gliner[structure]
result = model[text]
product = result.fetch("product").first

product["name"].text
# => "iPhone 15 Pro Max"

product["storage"].text
# => "256GB"

product["processor"].text
# => "A17 Pro chip"

product["price"].text
# => "$1199"
```
