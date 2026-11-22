from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class AddSubConfig:
    """Configuration for the independent Q-Mini Add/Sub v1 model.

    Defaults are a compact English-first model. They are not inherited from
    the legacy Q2-v2 configuration and can be changed only for a new run.
    """

    vocab_size: int = 512
    d_model: int = 96
    d_hidden: int = 170
    n_layers: int = 4
    n_experts: int = 2
    kernel_size: int = 7
    max_context: int = 512
    dropout: float = 0.05
    seed: int = 20261119

    def validate(self) -> None:
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "n_layers": self.n_layers,
            "n_experts": self.n_experts,
            "kernel_size": self.kernel_size,
            "max_context": self.max_context,
        }
        invalid = [name for name, value in positive.items() if value < 1]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {invalid}")
        if self.vocab_size < 264:
            raise ValueError("Byte-level BPE needs room for 256 byte symbols and eight required special tokens.")
        if self.d_model % 8:
            raise ValueError("d_model must be divisible by 8 for the reference integer path.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "AddSubConfig":
        if not isinstance(values, dict):
            raise TypeError("Configuration must be a dictionary.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"Unknown Add/Sub configuration fields: {unknown}")
        config = cls(**values)
        config.validate()
        return config
