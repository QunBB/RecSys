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


@dataclass
class Field:
    name: str
    emb: str = ""  # embedding name. If it is empty, it will be same as `name`
    dim: int = 4  # embedding size
    vocabulary_size: int = 1  # 0 or 1 for dense field, 0 is meant to not use embedding
    l2_reg: float = 0.  # embeddings l2 regularizer
    initializer: Union[str, Initializer] = "uniform"  # embeddings initializer
    belong: str = "user"  # what kind of the field
    length: int = 0  # history's max length, or dense field's dimension which don't use embedding
    group: str = "default"  # you can set different groups for multi domain or multitask or multi history
    dtype: str = "int32"

    def __post_init__(self):
        """
        history: user history behavior sequence
        user: user profile, like age, gender, etc.
        item: target item feature, like item_id, category_id, etc.
        domain: domain-side feature, like domain_id, statistics in special domain, etc.
        context: other context feature whose embeddings are usually concatenated directly as deep layer inputs
        task: task-side feature, like task_id, statistics in special task, etc.
        """
        assert self.belong in ["history", "user", "item", "domain", "context", "task"], f"Invalid Field.belong: \"{self.belong}\""

        if not self.emb:
            self.emb = self.name
