"""Auditable PyTorch implementation of the EPCDA network.

The model deliberately keeps the standardized network input ``x`` separate from
``rpm_raw``.  Mechanical phase is integrated from the latter only.  The default
experiment uses 128-sample records, 64-sample order-analysis windows, a hop of
16 samples, 16 learnable mechanical orders, and reflection smoothing with
radius two.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, coefficient: float) -> Tensor:
        ctx.coefficient = float(coefficient)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        return -ctx.coefficient * grad_output, None


def grad_reverse(x: Tensor, coefficient: float = 1.0) -> Tensor:
    """Identity in the forward pass and ``-coefficient`` in the backward pass."""

    return _GradientReversal.apply(x, coefficient)


def centered_cross_covariance_loss(z_f: Tensor, z_p: Tensor) -> Tensor:
    """Squared Frobenius norm of the centered empirical cross-covariance.

    This is the manuscript's orthogonality penalty, not an uncentered dot
    product.  A one-sample batch has no empirical covariance and returns a
    differentiable zero.
    """

    if z_f.ndim != 2 or z_p.ndim != 2:
        raise ValueError("z_f and z_p must both have shape [batch, features]")
    if z_f.shape[0] != z_p.shape[0]:
        raise ValueError("z_f and z_p must have the same batch size")
    batch_size = z_f.shape[0]
    if batch_size < 2:
        return (z_f.sum() + z_p.sum()) * 0.0
    f_centered = z_f - z_f.mean(dim=0, keepdim=True)
    p_centered = z_p - z_p.mean(dim=0, keepdim=True)
    cross_covariance = f_centered.transpose(0, 1) @ p_centered
    cross_covariance = cross_covariance / float(batch_size - 1)
    return cross_covariance.square().sum()


# Convenient name for training scripts and ablation code.
cross_covariance_loss = centered_cross_covariance_loss


def conditional_domain_features(
    z_f: Tensor, class_logits: Tensor, detach_probabilities: bool = True
) -> Tensor:
    """Return the multilinear feature/class input normally used by CDAN.

    The returned tensor has shape ``[B, num_classes * latent_dim]``.  Keeping
    this construction outside :class:`EPCDAModel` lets a training script swap a
    DANN head for a CDAN head without changing the representation network.
    """

    if z_f.ndim != 2 or class_logits.ndim != 2:
        raise ValueError("z_f and class_logits must both be rank-two tensors")
    if z_f.shape[0] != class_logits.shape[0]:
        raise ValueError("z_f and class_logits must have the same batch size")
    probabilities = class_logits.softmax(dim=-1)
    if detach_probabilities:
        probabilities = probabilities.detach()
    multilinear = torch.bmm(probabilities.unsqueeze(2), z_f.unsqueeze(1))
    return multilinear.flatten(start_dim=1)


class CDANDomainHead(nn.Module):
    """Optional conditional domain discriminator for matched CDAN baselines."""

    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        num_domains: int = 2,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        input_dim = int(latent_dim) * int(num_classes)
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(
        self,
        z_f: Tensor,
        class_logits: Tensor,
        grl_lambda: float = 1.0,
        detach_probabilities: bool = True,
    ) -> Tensor:
        features = conditional_domain_features(
            z_f, class_logits, detach_probabilities=detach_probabilities
        )
        return self.discriminator(grad_reverse(features, grl_lambda))


class OrderTrackedFrontEnd(nn.Module):
    """Learnable, mechanical-phase-locked complex order front end.

    Parameters are bounded through smooth transforms: center orders remain in
    [0.25, 6.0], Gaussian widths remain in [2, W/2] samples, and time offsets
    remain within one quarter-window.  Channel 0 is the fixed phase reference;
    the remaining channels have order-specific relative phase compensations in
    [-pi, pi].  This reference constraint removes the common-phase gauge that
    would otherwise cancel from both magnitude and cross-channel coherence.
    Each Gaussian envelope has unit L2 norm; multiplying it by a unit-magnitude
    complex oscillator therefore also yields a unit-L2 complex kernel.
    """

    _VALID_MODES = {"energy_only", "full"}

    def __init__(
        self,
        in_channels: int = 3,
        input_length: int = 128,
        num_orders: int = 16,
        window_size: int = 64,
        hop: int = 16,
        smoothing_radius: int = 2,
        mode: str = "full",
        min_order: float = 0.25,
        max_order: float = 6.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        mode = {"energy": "energy_only", "amplitude": "energy_only"}.get(mode, mode)
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(self._VALID_MODES)}")
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
        if input_length < window_size:
            raise ValueError("input_length must be at least window_size")
        if window_size < 2 or hop < 1:
            raise ValueError("window_size must exceed one and hop must be positive")
        if not 0 <= smoothing_radius:
            raise ValueError("smoothing_radius must be non-negative")
        if not min_order < max_order:
            raise ValueError("min_order must be smaller than max_order")

        self.in_channels = int(in_channels)
        self.input_length = int(input_length)
        self.num_orders = int(num_orders)
        self.window_size = int(window_size)
        self.hop = int(hop)
        self.smoothing_radius = int(smoothing_radius)
        self.mode = mode
        self.min_order = float(min_order)
        self.max_order = float(max_order)
        self.min_sigma = 2.0
        self.max_sigma = float(window_size) / 2.0
        self.eps = float(eps)

        # Initial values avoid saturated logits while covering the requested
        # physical range almost completely.
        initial_orders = torch.linspace(min_order + 0.05, max_order - 0.05, num_orders)
        order_fraction = (initial_orders - min_order) / (max_order - min_order)
        self.order_logits = nn.Parameter(torch.logit(order_fraction))

        initial_sigma = min(max(float(window_size) / 6.0, self.min_sigma + 0.1), self.max_sigma - 0.1)
        sigma_fraction = (initial_sigma - self.min_sigma) / (self.max_sigma - self.min_sigma)
        self.sigma_logits = nn.Parameter(
            torch.full((num_orders,), torch.logit(torch.tensor(sigma_fraction)).item())
        )
        self.offset_logits = nn.Parameter(torch.zeros(num_orders))
        # A phase shared by every channel is unidentifiable here: it disappears
        # from |Z_m| and from Z_a Z_b*.  Fix channel 0 (mechanical speed) as the
        # reference and learn only current-channel phases relative to it.
        self.phase_logits = nn.Parameter(torch.zeros(in_channels - 1, num_orders))

        pair_indices = torch.combinations(torch.arange(in_channels), r=2)
        self.register_buffer("pair_indices", pair_indices, persistent=False)

    @property
    def num_frames(self) -> int:
        return 1 + (self.input_length - self.window_size) // self.hop

    @property
    def num_pairs(self) -> int:
        return self.in_channels * (self.in_channels - 1) // 2

    @property
    def output_channels(self) -> int:
        base = self.in_channels
        if self.mode == "full":
            base += 2 * self.num_pairs
        return base * self.num_orders

    def orders(self) -> Tensor:
        return self.min_order + (self.max_order - self.min_order) * self.order_logits.sigmoid()

    def sigmas(self) -> Tensor:
        return self.min_sigma + (self.max_sigma - self.min_sigma) * self.sigma_logits.sigmoid()

    def offsets(self) -> Tensor:
        return (float(self.window_size) / 4.0) * self.offset_logits.tanh()

    def phase_offsets(self) -> Tensor:
        relative = math.pi * self.phase_logits.tanh()
        reference = relative.new_zeros((1, self.num_orders))
        return torch.cat((reference, relative), dim=0)

    def gaussian_windows(self) -> Tensor:
        """Return the unit-L2 Gaussian envelope for every learned order."""

        dtype = self.sigma_logits.dtype
        device = self.sigma_logits.device
        positions = torch.arange(self.window_size, dtype=dtype, device=device)
        center = (float(self.window_size) - 1.0) / 2.0 + self.offsets()
        gaussian = torch.exp(
            -0.5
            * ((positions.unsqueeze(0) - center.unsqueeze(1)) / self.sigmas().unsqueeze(1)).square()
        )
        return F.normalize(gaussian, p=2.0, dim=-1, eps=self.eps)

    @staticmethod
    def _batch_sampling_rate(
        fs: Union[float, Tensor], batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        value = torch.as_tensor(fs, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch_size)
        else:
            value = value.reshape(-1)
            if value.numel() == 1:
                value = value.expand(batch_size)
            elif value.numel() != batch_size:
                raise ValueError("fs must be scalar or contain one value per batch item")
        if torch.any(value <= 0):
            raise ValueError("all sampling rates must be positive")
        return value

    def mechanical_phase(self, rpm_raw: Tensor, fs: Union[float, Tensor]) -> Tensor:
        """Integrate unstandardized rpm into mechanical phase in radians."""

        if rpm_raw.ndim == 3 and rpm_raw.shape[1] == 1:
            rpm_raw = rpm_raw[:, 0]
        if rpm_raw.ndim != 2:
            raise ValueError("rpm_raw must have shape [B, L] or [B, 1, L]")
        batch_size, length = rpm_raw.shape
        sampling_rate = self._batch_sampling_rate(
            fs, batch_size, device=rpm_raw.device, dtype=rpm_raw.dtype
        )
        increments = (2.0 * math.pi / 60.0) * rpm_raw / sampling_rate.unsqueeze(1)
        phase = rpm_raw.new_zeros((batch_size, length))
        if length > 1:
            phase[:, 1:] = torch.cumsum(increments[:, :-1], dim=-1)
        return phase

    @staticmethod
    def _reflection_indices(length: int, radius: int, device: torch.device) -> Tensor:
        if length < 1:
            raise ValueError("cannot smooth an empty sequence")
        center = torch.arange(length, device=device).unsqueeze(1)
        offsets = torch.arange(-radius, radius + 1, device=device).unsqueeze(0)
        indices = center + offsets
        if length == 1:
            return torch.zeros_like(indices)
        period = 2 * (length - 1)
        indices = torch.remainder(indices, period)
        return torch.where(indices < length, indices, period - indices)

    def _reflect_smooth(self, values: Tensor) -> Tensor:
        radius = self.smoothing_radius
        if radius == 0:
            return values
        length = values.shape[-1]
        indices = self._reflection_indices(length, radius, values.device)
        selected = values.index_select(-1, indices.reshape(-1))
        selected = selected.reshape(*values.shape[:-1], length, 2 * radius + 1)
        return selected.mean(dim=-1)

    def forward(
        self, x: Tensor, rpm_raw: Tensor, fs: Union[float, Tensor]
    ) -> Dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, C, L]")
        batch_size, channels, length = x.shape
        if channels != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, received {channels}")
        if length != self.input_length:
            raise ValueError(f"expected input length {self.input_length}, received {length}")
        if rpm_raw.shape[0] != batch_size or rpm_raw.shape[-1] != length:
            raise ValueError("rpm_raw must match x in batch size and sample length")

        phase = self.mechanical_phase(rpm_raw.to(device=x.device, dtype=x.dtype), fs)
        x_frames = x.unfold(-1, self.window_size, self.hop)  # [B, C, T, W]
        phase_frames = phase.unfold(-1, self.window_size, self.hop)  # [B, T, W]
        local_phase = phase_frames - phase_frames[..., :1]

        orders = self.orders().to(dtype=x.dtype)
        sigmas = self.sigmas().to(dtype=x.dtype)
        gaussian = self.gaussian_windows().to(dtype=x.dtype)
        phase_offset = self.phase_offsets().to(dtype=x.dtype)
        argument = (
            local_phase[:, None, None, :, :] * orders[None, None, :, None, None]
            + phase_offset[None, :, :, None, None]
        )
        weighted_signal = x_frames[:, :, None, :, :] * gaussian[None, None, :, None, :]
        response_real = (weighted_signal * torch.cos(argument)).sum(dim=-1)
        response_imag = -(weighted_signal * torch.sin(argument)).sum(dim=-1)
        response = torch.complex(response_real, response_imag)  # [B, C, K, T]

        amplitude = torch.sqrt(response.real.square() + response.imag.square() + self.eps)
        feature_parts = [amplitude.reshape(batch_size, channels * self.num_orders, -1)]

        coherence_values = []
        for pair in self.pair_indices.tolist():
            a, b = int(pair[0]), int(pair[1])
            z_a = response[:, a]
            z_b = response[:, b]
            cross = self._reflect_smooth(z_a * z_b.conj())
            auto_a = self._reflect_smooth(z_a.abs().square())
            auto_b = self._reflect_smooth(z_b.abs().square())
            denominator = torch.sqrt(auto_a * auto_b + self.eps)
            coherence_values.append(cross / (denominator + self.eps))

        if coherence_values:
            coherence = torch.stack(coherence_values, dim=1)  # [B, pairs, K, T]
        else:
            coherence = response.new_empty(
                (batch_size, 0, self.num_orders, response.shape[-1])
            )

        if self.mode == "full":
            feature_parts.append(coherence.real.reshape(batch_size, self.num_pairs * self.num_orders, -1))
            feature_parts.append(coherence.imag.reshape(batch_size, self.num_pairs * self.num_orders, -1))
        features = torch.cat(feature_parts, dim=1)

        return {
            "features": features,
            "amplitude": amplitude,
            "coherence": coherence,
            "response": response,
            "phase": phase,
            "orders": orders,
            "sigmas": sigmas,
            "phase_offsets": phase_offset,
            "gaussian": gaussian,
        }


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class GlobalLocalEncoder(nn.Module):
    """Small attention/global plus depthwise-convolution/local encoder."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        max_tokens: int = 128,
        stem_kernel: int = 1,
        stem_stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = stem_kernel // 2
        groups = _normalization_groups(hidden_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                hidden_dim,
                kernel_size=stem_kernel,
                stride=stem_stride,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, max_tokens, hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)

        num_heads = 4 if hidden_dim % 4 == 0 else 1
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.global_norm_1 = nn.LayerNorm(hidden_dim)
        self.global_ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.global_norm_2 = nn.LayerNorm(hidden_dim)

        self.local_block = nn.Sequential(
            nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False
            ),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.global_gate = nn.Linear(hidden_dim, hidden_dim)
        self.local_gate = nn.Linear(hidden_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _positional_embedding(self, tokens: int) -> Tensor:
        if tokens <= self.position.shape[1]:
            return self.position[:, :tokens]
        interpolated = F.interpolate(
            self.position.transpose(1, 2), size=tokens, mode="linear", align_corners=False
        )
        return interpolated.transpose(1, 2)

    def forward(self, features: Tensor) -> Tensor:
        sequence = self.stem(features)
        tokens = sequence.transpose(1, 2)
        tokens = tokens + self._positional_embedding(tokens.shape[1]).to(dtype=tokens.dtype)

        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        global_sequence = self.global_norm_1(tokens + attended)
        global_sequence = self.global_norm_2(
            global_sequence + self.global_ffn(global_sequence)
        )

        local_sequence = sequence + self.local_block(sequence)
        global_context = global_sequence.mean(dim=1)
        local_context = local_sequence.mean(dim=-1)
        global_context = global_context * torch.sigmoid(self.global_gate(local_context))
        local_context = local_context * torch.sigmoid(self.local_gate(global_context))
        return self.fusion(torch.cat((global_context, local_context), dim=-1))


class _MLPHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[int] = None) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(input_dim // 2, output_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class EPCDAModel(nn.Module):
    """End-to-end EPCDA representation, classifier, and adversarial heads.

    ``x`` must already be standardized with source-training statistics.  In
    contrast, ``rpm_raw`` must remain in physical r/min and is used only for
    phase integration.  ``raw_encoder=True`` supplies the A0/shared-backbone
    alternative without evaluating the order front end.
    """

    def __init__(
        self,
        num_classes: int = 5,
        num_domains: int = 2,
        num_powers: int = 3,
        in_channels: int = 3,
        input_length: int = 128,
        num_orders: int = 16,
        window_size: int = 64,
        hop: int = 16,
        smoothing_radius: int = 2,
        front_mode: str = "full",
        raw_encoder: bool = False,
        hidden_dim: int = 64,
        representation_dim: int = 128,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if front_mode == "raw":
            raw_encoder = True
            front_mode = "full"
        front_mode = {"energy": "energy_only", "amplitude": "energy_only"}.get(
            front_mode, front_mode
        )
        self.raw_encoder = bool(raw_encoder)
        self.front_mode = front_mode
        self.input_length = int(input_length)

        self.front_end = OrderTrackedFrontEnd(
            in_channels=in_channels,
            input_length=input_length,
            num_orders=num_orders,
            window_size=window_size,
            hop=hop,
            smoothing_radius=smoothing_radius,
            mode=front_mode,
        )
        if self.raw_encoder:
            encoder_input_channels = in_channels
            stem_kernel, stem_stride = 7, 4
            max_tokens = int(math.ceil(input_length / stem_stride))
        else:
            encoder_input_channels = self.front_end.output_channels
            stem_kernel, stem_stride = 1, 1
            max_tokens = self.front_end.num_frames

        self.encoder = GlobalLocalEncoder(
            input_channels=encoder_input_channels,
            hidden_dim=hidden_dim,
            output_dim=representation_dim,
            max_tokens=max_tokens,
            stem_kernel=stem_kernel,
            stem_stride=stem_stride,
            dropout=dropout,
        )
        self.fault_projector = nn.Sequential(
            nn.Linear(representation_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU()
        )
        self.power_projector = nn.Sequential(
            nn.Linear(representation_dim, latent_dim), nn.LayerNorm(latent_dim), nn.GELU()
        )
        self.classifier = nn.Linear(latent_dim, num_classes)
        self.domain_discriminator = _MLPHead(latent_dim, num_domains)
        self.power_predictor_p = _MLPHead(latent_dim, num_powers)
        self.power_predictor_f = _MLPHead(latent_dim, num_powers)

    def forward(
        self,
        x: Tensor,
        rpm_raw: Tensor,
        fs: Union[float, Tensor],
        grl_lambda: float = 1.0,
        power_grl_lambda: Optional[float] = None,
    ) -> Dict[str, Optional[Tensor]]:
        if x.ndim != 3 or x.shape[-1] != self.input_length:
            raise ValueError(f"x must have shape [B, C, {self.input_length}]")
        power_grl_lambda = grl_lambda if power_grl_lambda is None else power_grl_lambda

        front: Optional[Dict[str, Tensor]]
        if self.raw_encoder:
            front = None
            encoder_input = x
            orders = self.front_end.orders().to(device=x.device, dtype=x.dtype)
            sigmas = self.front_end.sigmas().to(device=x.device, dtype=x.dtype)
        else:
            front = self.front_end(x, rpm_raw, fs)
            encoder_input = front["features"]
            orders = front["orders"]
            sigmas = front["sigmas"]

        h = self.encoder(encoder_input)
        z_f = self.fault_projector(h)
        z_p = self.power_projector(h)
        class_logits = self.classifier(z_f)

        # Exactly one GRL is used on each adversarial path.  The direct z_p
        # power head deliberately has none.
        domain_logits = self.domain_discriminator(grad_reverse(z_f, grl_lambda))
        power_p_logits = self.power_predictor_p(z_p)
        power_f_logits = self.power_predictor_f(grad_reverse(z_f, power_grl_lambda))

        output: Dict[str, Optional[Tensor]] = {
            "h": h,
            "zf": z_f,
            "zp": z_p,
            "cls": class_logits,
            "dom": domain_logits,
            "powp": power_p_logits,
            "powf": power_f_logits,
            "orders": orders,
            "sigmas": sigmas,
            "front_features": None if front is None else front["features"],
            "amplitude": None if front is None else front["amplitude"],
            "coherence": None if front is None else front["coherence"],
            "response": None if front is None else front["response"],
            "phase": None if front is None else front["phase"],
        }
        return output

    @staticmethod
    def orthogonality_loss(z_f: Tensor, z_p: Tensor) -> Tensor:
        return centered_cross_covariance_loss(z_f, z_p)

    @staticmethod
    def cdan_features(
        z_f: Tensor, class_logits: Tensor, detach_probabilities: bool = True
    ) -> Tensor:
        return conditional_domain_features(z_f, class_logits, detach_probabilities)


# Short alias retained for experiment configuration files.
EPCDA = EPCDAModel


__all__ = [
    "CDANDomainHead",
    "EPCDA",
    "EPCDAModel",
    "GlobalLocalEncoder",
    "OrderTrackedFrontEnd",
    "centered_cross_covariance_loss",
    "conditional_domain_features",
    "cross_covariance_loss",
    "grad_reverse",
]
