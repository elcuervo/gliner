# Single entities

```ruby
text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."

entities = ["company", "person", "product", "location"]
model = Gliner[entities]
model[text]

# => {company"=>["Apple"], "person"=>["Tim Cook"], "product"=>["iPhone 15"], "location"=>["Cupertino"]}
```

# Per entity config

```ruby
text = "Email John Doe at john@example.com."

entities = {
  email: { description: "Email addresses", dtype: "list", threshold: 0.9 },
  person: { description: "Person names", dtype: "str" }
}

model = Gliner[entities]
model[text]

# => {company"=>["Apple"], "person"=>["Tim Cook"], "product"=>["iPhone 15"], "location"=>["Cupertino"]}
```

# Classfication

```ruby
text = "This laptop has amazing performance but terrible battery life!",
concept = { "sentiment" => %w[positive negative neutral] }
model = Gliner.classify[concept]
model[text]

# => {"sentiment"=>"negative"}
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
model[text]

# => {"product"=>[{"name"=>"iPhone 15 Pro Max", "storage"=>"256GB", "processor"=>"A17 Pro chip", "price"=>"$1199"}]}
```
