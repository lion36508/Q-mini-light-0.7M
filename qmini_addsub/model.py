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

    def step(
        self, inputs: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one token and retain only the causal convolution history."""
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("Convolution step expects [batch, 1, features].")
        padding = self.dilation * (self.kernel_size - 1)
        expected_history = (inputs.shape[0], padding, inputs.shape[2])
        if tuple(history.shape) != expected_history:
            raise ValueError(
                f"Convolution cache shape must be {expected_history}, got {tuple(history.shape)}."
            )
        joined = torch.cat((history, inputs), dim=1)
        mixed = F.conv1d(
            joined.transpose(1, 2),
            quantize_weight_ste(self.depthwise_weight),
            self.depthwise_bias,
            dilation=self.dilation,
            groups=inputs.shape[-1],
        ).transpose(1, 2)
        content, gate = self.gate_projection(mixed).chunk(2, dim=-1)
        activated = F.silu(content) * torch.sigmoid(gate)
        updated_history = joined[:, -padding:, :] if padding else joined[:, :0, :]
        return self.output_projection(activated), updated_history


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

    def step(
        self, inputs: torch.Tensor, previous_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one token using the diagonal recurrent state."""
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("State-space step expects [batch, 1, features].")
        if tuple(previous_state.shape) != (inputs.shape[0], inputs.shape[2]):
            raise ValueError("State-space cache must have shape [batch, features].")
        drive = torch.tanh(self.drive_projection(inputs)).float()[:, 0, :]
        decay = torch.exp(
            -F.softplus(self.decay_projection(inputs).float())
        ).clamp_min(1e-6)[:, 0, :]
        current_state = decay * previous_state.float() + (1.0 - decay) * drive
        gated = current_state.to(dtype=inputs.dtype).unsqueeze(1)
        gated = gated * torch.sigmoid(self.gate_projection(inputs))
        return self.output_projection(gated), current_state


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

    def forward(
        self,
        inputs: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(inputs)
        probabilities = torch.softmax(logits.float(), dim=-1)
        selected = torch.argmax(logits, dim=-1)
        selected_probability = probabilities.gather(-1, selected.unsqueeze(-1))

        if token_mask is None:
            valid_probabilities = probabilities.reshape(-1, self.n_experts)
            valid_selected = selected.reshape(-1)
        else:
            if token_mask.shape != inputs.shape[:2]:
                raise ValueError("MoE token_mask must match [batch, sequence] inputs.")
            valid = token_mask.to(device=inputs.device, dtype=torch.bool).reshape(-1)
            if not bool(valid.any()):
                raise ValueError("MoE token_mask must include at least one valid token.")
            valid_probabilities = probabilities.reshape(-1, self.n_experts)[valid]
            valid_selected = selected.reshape(-1)[valid]

        assignments = F.one_hot(valid_selected, num_classes=self.n_experts).float()
        load = assignments.mean(dim=0)
        importance = valid_probabilities.mean(dim=0)
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

    def forward(
        self,
        inputs: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conv_update = self.dropout(self.conv(self.conv_norm(inputs)))
        outputs = inputs + torch.sigmoid(self.conv_residual_logit) * conv_update

        state_update = self.dropout(self.state_space(self.state_norm(outputs)))
        outputs = outputs + torch.sigmoid(self.state_residual_logit) * state_update

        expert_output, auxiliary_loss = self.experts(
            self.expert_norm(outputs), token_mask=token_mask
        )
        outputs = outputs + torch.sigmoid(self.expert_residual_logit) * self.dropout(expert_output)
        return outputs, auxiliary_loss

    def step(
        self,
        inputs: torch.Tensor,
        cache: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run one token through this block and return its updated local caches."""
        conv_history, state = cache
        conv_update, conv_history = self.conv.step(
            self.conv_norm(inputs), conv_history
        )
        outputs = inputs + torch.sigmoid(self.conv_residual_logit) * conv_update

        state_update, state = self.state_space.step(
            self.state_norm(outputs), state
        )
        outputs = outputs + torch.sigmoid(self.state_residual_logit) * state_update

        expert_output, _ = self.experts(self.expert_norm(outputs))
        outputs = outputs + torch.sigmoid(self.expert_residual_logit) * expert_output
        return outputs, (conv_history, state)


class AddSubLanguageModel(nn.Module):
    """Independent random-initialized Add/Sub language model for pretraining."""

    def __init__(
        self,
        config: AddSubConfig | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AddSubConfig()
        self.config.validate()
        model_seed = self.config.seed if seed is None else seed
        if isinstance(model_seed, bool) or not isinstance(model_seed, int) or model_seed < 0:
            raise ValueError("Model seed must be a non-negative integer.")
        # Isolate parameter initialization so constructing a model does not
        # consume or reset the caller's training/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(model_seed)
            self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        initialize_five_symbol_weight_(self.token_embedding.weight)
        self.blocks = nn.ModuleList(
            [
                AddSubBlock(self.config, dilation=2 ** index)
                for index in range(self.config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(self.config.d_model)
        self.final_bias = nn.Parameter(torch.zeros(self.config.vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have non-empty [batch, sequence] shape.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer torch.int32 or torch.int64 dtype.")
        if input_ids.shape[1] > self.config.max_context:
            raise ValueError("Sequence exceeds configured maximum context.")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape.")
            if attention_mask.device != input_ids.device:
                raise ValueError("attention_mask and input_ids must be on the same device.")
            attention_mask = attention_mask.to(dtype=torch.bool)
            if not bool(attention_mask.any()):
                raise ValueError("attention_mask must include at least one valid token.")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("Token ID is outside the configured vocabulary.")

        embedding_weight = quantize_weight_ste(self.token_embedding.weight)
        hidden = F.embedding(input_ids, embedding_weight)
        auxiliary_losses = []
        for block in self.blocks:
            hidden, aux_loss = block(hidden, token_mask=attention_mask)
            auxiliary_losses.append(aux_loss)

        tied_output_weight = quantize_weight_ste(self.token_embedding.weight)
        logits = F.linear(self.final_norm(hidden), tied_output_weight, self.final_bias)
        auxiliary_loss = torch.stack(auxiliary_losses).mean()
        return {"logits": logits, "aux_loss": auxiliary_loss}

    def init_cache(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Create independent convolution and recurrent-state caches per block."""
        if isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        parameter = self.token_embedding.weight
        target_device = parameter.device if device is None else torch.device(device)
        dtype = parameter.dtype
        caches = []
        for block in self.blocks:
            padding = block.conv.dilation * (block.conv.kernel_size - 1)
            history = torch.zeros(
                batch_size, padding, self.config.d_model,
                device=target_device, dtype=dtype,
            )
            state = torch.zeros(
                batch_size, self.config.d_model,
                device=target_device, dtype=torch.float32,
            )
            caches.append((history, state))
        return caches

    def forward_step(
        self,
        input_ids: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Evaluate one token per batch item with bounded sequence caches.

        This is a streaming reference path for the PyTorch graph. It does not
        claim the separate integer-only deployed runtime contract.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before using forward_step().")
        if input_ids.ndim != 1:
            raise ValueError("forward_step input_ids must have shape [batch].")
        if len(caches) != len(self.blocks):
            raise ValueError("One convolution/state cache pair is required per block.")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("Token ID is outside the configured vocabulary.")

        embedding_weight = quantize_weight_ste(self.token_embedding.weight)
        hidden = F.embedding(input_ids, embedding_weight).unsqueeze(1)
        updated_caches = []
        for block, cache in zip(self.blocks, caches):
            hidden, updated = block.step(hidden, cache)
            updated_caches.append(updated)
        hidden = self.final_norm(hidden[:, 0, :])
        tied_output_weight = quantize_weight_ste(self.token_embedding.weight)
        logits = F.linear(hidden, tied_output_weight, self.final_bias)
        return logits, updated_caches

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
