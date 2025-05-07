from typing import Union
from dataclasses import dataclass

from tensorflow.keras.initializers import Initializer

@dataclass
class Task:
    name: str
    belong: str = "binary"
    num_classes: int = 1  # just for multiclass
    return_logit: bool = False  # whether to return logit for ranking loss

    def __post_init__(self):
        assert self.belong in ["binary", "regression", "multiclass"], f"Invalid Task.belong: \"{self.belong}\""

        self.return_logit = self.belong == "binary" and self.return_logit

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash((self.name, self.belong, self.num_classes))


@dataclass
class Field:
    name: str
    emb: str = ""  # embedding name. If it is empty, it will be same as `name`
    dim: int = 4  # embedding size
    vocabulary_size: int = 1  # 0 or 1 for dense field, 0 is meant to not use embedding
    l2_reg: float = 0.  # embeddings l2 regularizer
    initializer: Union[str, Initializer] = "uniform"  # embeddings initializer
    belong: str = "user"  # what kind of the field
    length: int = 1  # history's max length, or dense field's dimension which don't use embedding
    group: str = "default"  # you can set different groups for multitask or multi history or field groups
    dtype: str = "int32"

    def __post_init__(self):
        """
        history: user history behavior sequence.
        user: user profile, e.g., age, gender.
        item: target item feature, e.g., item_id, category_id.
        domain: domain-side feature, e.g., domain_id, statistics in special domain.
        context: other context feature whose embeddings are usually concatenated directly as deep layer inputs, e.g., timestamp.
        task: task-side feature, e.g., task_id, statistics in special task.
        """
        assert self.belong in ["history", "user", "item", "domain", "context", "task"], f"Invalid Field.belong: \"{self.belong}\""

        if not self.emb:
            self.emb = self.name
