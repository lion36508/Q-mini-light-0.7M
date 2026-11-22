from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AddSubConfig
from .quantization import initialize_five_symbol_weight_, quantize_weight_ste


class AddSubLinear(nn.Linear):
    """Training projection with a five-symbol STE weight codebook.

    This module belongs to the training graph. It uses PyTorch autograd and
    floating-point arithmetic during optimization. The deployed inference
    graph must use qmini_addsub.runtime and its integer lookup/add path.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        initialize_five_symbol_weight_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, quantize_weight_ste(self.weight), self.bias)


class CausalConvMixer(nn.Module):
    def __init__(self, config: AddSubConfig) -> None:
        super().__init__()
        self.kernel_size = config.kernel_size
        self.dilation = 1
        self.depthwise_weight = nn.Parameter(
            torch.empty(config.d_model, 1, config.kernel_size)
        )
        self.depthwise_bias = nn.Parameter(torch.zeros(config.d_model))
        self.gate_projection = AddSubLinear(config.d_model, config.d_model * 2)
        self.output_projection = AddSubLinear(config.d_model, config.d_model)
        initialize_five_symbol_weight_(self.depthwise_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel_first = inputs.transpose(1, 2)
        padding = self.dilation * (self.kernel_size - 1)
        padded = F.pad(channel_first, (padding, 0))
        mixed = F.conv1d(
            padded,
            quantize_weight_ste(self.depthwise_weight),
            self.depthwise_bias,
            dilation=self.dilation,
            groups=inputs.shape[-1],
        ).transpose(1, 2)
        content, gate = self.gate_projection(mixed).chunk(2, dim=-1)
        activated = F.silu(content) * torch.sigmoid(gate)
        return self.output_projection(activated)


class RMSNorm(nn.Module):
    """Pre-normalization for stable residual streams."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight.to(inputs.dtype)


class GatedDiagonalStateSpace(nn.Module):
    """Input-dependent diagonal state-space memory with an associative scan."""

    def __init__(self, config: AddSubConfig) -> None:
        super().__init__()
        self.drive_projection = AddSubLinear(config.d_model, config.d_model)
        self.decay_projection = AddSubLinear(config.d_model, config.d_model)
        self.gate_projection = AddSubLinear(config.d_model, config.d_model)
        self.output_projection = AddSubLinear(config.d_model, config.d_model)
        with torch.no_grad():
            log_times = torch.linspace(
                torch.log(torch.tensor(2.0)),
                torch.log(torch.tensor(float(config.max_context))),
                config.d_model,
            )
            time_constants = torch.exp(log_times)
            rates = time_constants.reciprocal()
            self.decay_projection.weight.zero_()
            self.decay_projection.bias.copy_(torch.log(torch.expm1(rates)))

    @staticmethod
    def parallel_scan(decay: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        """Differentiable O(log(sequence))-depth affine prefix scan."""
        if decay.ndim != 3 or injection.shape != decay.shape:
            raise ValueError("decay and injection must share [batch, sequence, features] shape.")
        if decay.shape[1] < 1:
            raise ValueError("State-space scan requires a non-empty sequence.")
        a = decay.float()
        b = injection.float()
        offset, length = 1, decay.shape[1]
        while offset < length:
            previous_a, previous_b = a, b
            earlier_a = torch.cat(
                (torch.ones_like(previous_a[:, :offset]), previous_a[:, :-offset]),
                dim=1,
            )
            earlier_b = torch.cat(
                (torch.zeros_like(previous_b[:, :offset]), previous_b[:, :-offset]),
                dim=1,
            )
            b = previous_b + previous_a * earlier_b
            a = previous_a * earlier_a
            offset *= 2
        return b

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        drive = torch.tanh(self.drive_projection(inputs)).float()
        decay = torch.exp(-F.softplus(self.decay_projection(inputs).float())).clamp_min(1e-6)
        states = self.parallel_scan(decay, (1.0 - decay) * drive)
        gated = states.to(dtype=inputs.dtype) * torch.sigmoid(self.gate_projection(inputs))
        return self.output_projection(gated)


class SwiGLUExpert(nn.Module):
    """Parameter-efficient gated expert MLP used by the sparse router."""

    def __init__(self, config: AddSubConfig) -> None:
        super().__init__()
        self.up_gate = AddSubLinear(config.d_model, config.d_hidden * 2)
        self.down = AddSubLinear(config.d_hidden, config.d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        content, gate = self.up_gate(inputs).chunk(2, dim=-1)
        return self.down(F.silu(content) * gate)


class SparseTop1Experts(nn.Module):
    def __init__(self, config: AddSubConfig) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.router = AddSubLinear(config.d_model, config.n_experts)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(config) for _ in range(config.n_experts)]
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(inputs)
        probabilities = torch.softmax(logits.float(), dim=-1)
        selected = torch.argmax(logits, dim=-1)
        selected_probability = probabilities.gather(-1, selected.unsqueeze(-1))

        assignments = F.one_hot(selected, num_classes=self.n_experts).float()
        load = assignments.mean(dim=(0, 1))
        importance = probabilities.mean(dim=(0, 1))
        auxiliary_loss = self.n_experts * torch.sum(load * importance)

        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_selected = selected.reshape(-1)
        flat_probability = selected_probability.reshape(-1, 1)
        output = torch.zeros_like(flat_inputs)
        for expert_index, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                flat_selected == expert_index, as_tuple=False
            ).flatten()
            if token_indices.numel() == 0:
                continue
            expert_inputs = flat_inputs.index_select(0, token_indices)
            expert_probability = flat_probability.index_select(0, token_indices)
            expert_output = expert(expert_inputs) * expert_probability
            output = output.index_copy(0, token_indices, expert_output)
        return output.reshape_as(inputs), auxiliary_loss


class AddSubBlock(nn.Module):
    """Pre-norm residual block: causal convolution, state-space memory and sparse MoE."""

    def __init__(self, config: AddSubConfig, dilation: int = 1) -> None:
        super().__init__()
        self.conv_norm = RMSNorm(config.d_model)
        self.conv = CausalConvMixer(config)
        self.conv.dilation = dilation
        self.state_norm = RMSNorm(config.d_model)
        self.state_space = GatedDiagonalStateSpace(config)
        self.expert_norm = RMSNorm(config.d_model)
        self.experts = SparseTop1Experts(config)
        self.dropout = nn.Dropout(config.dropout)
        # Bounded, learnable branch strengths stabilize the residual stream
        # without hard clipping activations or zeroing gradients at a threshold.
        self.conv_residual_logit = nn.Parameter(torch.zeros(()))
        self.state_residual_logit = nn.Parameter(torch.zeros(()))
        self.expert_residual_logit = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        conv_update = self.dropout(self.conv(self.conv_norm(inputs)))
        outputs = inputs + torch.sigmoid(self.conv_residual_logit) * conv_update

        state_update = self.dropout(self.state_space(self.state_norm(outputs)))
        outputs = outputs + torch.sigmoid(self.state_residual_logit) * state_update

        expert_output, auxiliary_loss = self.experts(self.expert_norm(outputs))
        outputs = outputs + torch.sigmoid(self.expert_residual_logit) * self.dropout(expert_output)
        return outputs, auxiliary_loss


class AddSubLanguageModel(nn.Module):
    """Independent random-initialized Add/Sub language model for pretraining."""

    def __init__(self, config: AddSubConfig | None = None) -> None:
        super().__init__()
        self.config = config or AddSubConfig()
        self.config.validate()
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        initialize_five_symbol_weight_(self.token_embedding.weight)
        self.blocks = nn.ModuleList(
            [AddSubBlock(self.config, dilation=2 ** index) for index in range(self.config.n_layers)]
        )
        self.final_norm = RMSNorm(self.config.d_model)
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
        logits = F.linear(self.final_norm(hidden), tied_output_weight, self.final_bias)
        auxiliary_loss = torch.stack(auxiliary_losses).mean()
        return {"logits": logits, "aux_loss": auxiliary_loss}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
