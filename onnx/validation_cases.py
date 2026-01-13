from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple


@dataclass(frozen=True)
class EntitiesCase:
    text: str
    labels: Sequence[str]
    descriptions: Mapping[str, str]
    threshold: float = 0.5


@dataclass(frozen=True)
class ClassificationCase:
    text: str
    task_name: str
    labels: Sequence[str]
    threshold: float = 0.5
    multi_label: bool = False


@dataclass(frozen=True)
class JsonCase:
    text: str
    parent: str
    fields: Sequence[str]
    threshold: float = 0.5


def build_validation_cases() -> Tuple[EntitiesCase, ClassificationCase, JsonCase]:
    return (
        EntitiesCase(
            text="Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday.",
            labels=["company", "person", "product", "location"],
            descriptions={"company": "Organization or business names"},
        ),
        ClassificationCase(
            text="This laptop has amazing performance but terrible battery life!",
            task_name="sentiment",
            labels=["positive", "negative", "neutral"],
        ),
        JsonCase(
            text="iPhone 15 Pro Max with 256GB storage, priced at $1199.",
            parent="product",
            fields=[
                "name::str::Full product name and model",
                "storage::str::Storage capacity",
                "price::str::Product price with currency",
            ],
        ),
    )
