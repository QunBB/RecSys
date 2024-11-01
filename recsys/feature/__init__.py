from dataclasses import dataclass


@dataclass
class Task:
    name: str
    belong: str = "binary"
    num_classes: int = 1  # just for multiclass

    def __post_init__(self):
        assert self.belong in ["binary", "regression", "multiclass"], f"Invalid Task.belong: {self.belong}"


@dataclass
class Field:
    name: str
    emb: str = ""  # embedding name
    dim: int = 4  # embedding size
    vocabulary_size: int = 1  # 0 or 1 for dense field, 0 is meant to not use embedding
    l2_reg: float = 0.  # embeddings l2 regularizer
    initializer: str = "glorot_normal"  # embeddings initializer
    belong: str = "user"  # what kind of the field
    length: int = 0  # history's max length, or dense field's dimension which don't use embedding
    group: str = "default"  # you can set different groups for multi domain or multi history
    dtype: str = "int32"

    def __post_init__(self):
        assert self.belong in ["history", "user", "item", "domain", "context"], f"Invalid Field.belong: {self.belong}"
