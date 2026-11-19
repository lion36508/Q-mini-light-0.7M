from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AddSubConfig
from .quantization import quantize_weight_ste


class AddSubLinear(nn.Linear):
    """Training projection with a five-symbol STE weight codebook.

    This module belongs to the training graph. It uses PyTorch autograd and
    floating-point arithmetic during optimization. The deployed inference
    graph must use qmini_addsub.runtime and its integer lookup/add path.
    """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, quantize_weight_ste(self.weight), self.bias)


class CausalConvMixer(nn.Module):
    def __init__(self, config: AddSubConfig) -> None:
        super().__init__()
        self.kernel_size = config.kernel_size
        self.depthwise_weight = nn.Parameter(
            torch.empty(config.d_model, 1, config.kernel_size)
        )
        self.depthwise_bias = nn.Parameter(torch.zeros(config.d_model))
        self.gate_projection = AddSubLinear(config.d_model, config.d_model * 2)
        self.output_projection = AddSubLinear(config.d_model, config.d_model)
        nn.init.normal_(self.depthwise_weight, mean=0.0, std=0.15)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel_first = inputs.transpose(1, 2)
        padded = F.pad(channel_first, (self.kernel_size - 1, 0))
        mixed = F.conv1d(
            padded,
            quantize_weight_ste(self.depthwise_weight),
            self.depthwise_bias,
            groups=inputs.shape[-1],
        ).transpose(1, 2)
        content, gate = self.gate_projection(mixed).chunk(2, dim=-1)
        activated = F.silu(content) * torch.sigmoid(gate)
        return self.output_projection(activated)


class GatedDiagonalStateSpace(nn.Module):
    """Causal diagonal recurrence with a parallelizable training interface."""

    def __init__(self, config: AddSubConfig) -> None:
        super().__init__()
        self.drive_projection = AddSubLinear(config.d_model, config.d_model)
        self.gate_projection = AddSubLinear(config.d_model, config.d_model)
        self.decay_logits = nn.Parameter(torch.zeros(config.d_model))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        drive = torch.tanh(self.drive_projection(inputs))
        decay = torch.sigmoid(self.decay_logits).view(1, -1)
        state = torch.zeros_like(drive[:, 0, :])
        states = []
        for position in range(drive.shape[1]):
            state = decay * state + (1.0 - decay) * drive[:, position, :]
            states.append(state)
        memory = torch.stack(states, dim=1)
        return memory * torch.sigmoid(self.gate_projection(inputs))


class SparseTop1Experts(nn.Module):
    def __init__(self, config: AddSubConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.router = AddSubLinear(config.d_model, config.n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    AddSubLinear(config.d_model, config.d_hidden),
                    nn.SiLU(),
                    AddSubLinear(config.d_hidden, config.d_model),
                )
                for _ in range(config.n_experts)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(inputs)
        probabilities = torch.softmax(logits.float(), dim=-1)
        selected = torch.argmax(logits, dim=-1)
        selected_probability = probabilities.gather(-1, selected.unsqueeze(-1))

        output = torch.zeros_like(inputs)
        assignments = F.one_hot(selected, num_classes=self.n_experts).float()
        load = assignments.mean(dim=(0, 1))
        importance = probabilities.mean(dim=(0, 1))
        auxiliary_loss = self.n_experts * torch.sum(load * importance)

        for expert_index, expert in enumerate(self.experts):
            expert_output = expert(inputs)
            mask = (selected == expert_index).unsqueeze(-1)
            weighted = expert_output * selected_probability
            output = output + torch.where(mask, weighted, torch.zeros_like(weighted))
        return output, auxiliary_loss


class AddSubBlock(nn.Module):
    """New implementation of the legacy model's three useful sequence ideas."""

    def __init__(self, config: AddSubConfig) -> None:
        super().__init__()
        self.conv = CausalConvMixer(config)
        self.state_space = GatedDiagonalStateSpace(config)
        self.experts = SparseTop1Experts(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = inputs + self.dropout(self.conv(inputs))
        outputs = outputs + self.dropout(self.state_space(outputs))
        expert_output, auxiliary_loss = self.experts(outputs)
        outputs = torch.clamp(outputs + self.dropout(expert_output), -12.0, 12.0)
        return outputs, auxiliary_loss


class AddSubLanguageModel(nn.Module):
    """Independent random-initialized Add/Sub language model for pretraining."""

    def __init__(self, config: AddSubConfig | None = None) -> None:
        super().__init__()
        self.config = config or AddSubConfig()
        self.config.validate()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.blocks = nn.ModuleList(
            [AddSubBlock(self.config) for _ in range(self.config.n_layers)]
        )
        self.final_bias = nn.Parameter(torch.zeros(self.config.vocab_size))

    def forward(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")
        if input_ids.shape[1] > self.config.max_context:
            raise ValueError("Sequence exceeds configured maximum context.")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("Token ID is outside the configured vocabulary.")

        embedding_weight = quantize_weight_ste(self.token_embedding.weight)
        hidden = F.embedding(input_ids, embedding_weight)
        auxiliary_losses = []
        for block in self.blocks:
            hidden, aux_loss = block(hidden)
            auxiliary_losses.append(aux_loss)

        tied_output_weight = quantize_weight_ste(self.token_embedding.weight)
        logits = F.linear(hidden, tied_output_weight, self.final_bias)
        auxiliary_loss = torch.stack(auxiliary_losses).mean()
        return {"logits": logits, "aux_loss": auxiliary_loss}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
