#!/usr/bin/env python3
"""Paper-aligned DMM-WcycleGAN model.

This file focuses on the model and training logic described in the paper:

1. Dual generators ``G1`` / ``G2`` for source-target bidirectional mapping
2. Dual WGAN critics ``D1`` / ``D2`` with gradient penalty
3. A universal online CNN discriminator ``Du`` for 3-class decoding
4. Three-stage training flow:
   - meta-initialization
   - fine-tuning with frozen lower convolution blocks
   - online classifier training on the augmented target-domain set

The paper does not publish every layer width in the main text, so the module
below implements a compact convolutional realization that preserves the paper's
described structure and training objectives.
"""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


LOGGER = logging.getLogger("dmm_wcyclegan")


@dataclass
class DMMWcycleGANConfig:
    feature_shape: tuple[int, int] = (80, 8)
    num_classes: int = 3
    generator_channels: int = 32
    critic_channels: int = 32
    classifier_channels: int = 32
    lambda_adv: float = 1.0
    lambda_cycle: float = 10.0
    lambda_identity: float = 5.0
    lambda_gp: float = 3.0
    lambda_reg: float = 1.0e-4
    max_meta_epochs: int = 20
    max_inner_steps: int = 4
    max_trans_epochs: int = 200
    max_cnn_epochs: int = 100
    meta_inner_lr: float = 1.0e-3
    meta_outer_step: float = 0.1
    fine_tune_lr: float = 1.0e-3
    critic_lr: float = 1.0e-3
    classifier_lr: float = 1.0e-3
    pruning_threshold: float = 1.0e-3
    prune_every: int = 10
    batch_size: int = 32


@dataclass
class MetaTask:
    support_source: Tensor
    support_target: Tensor
    query_source: Tensor
    query_target: Tensor

    def to(self, device: torch.device) -> "MetaTask":
        return MetaTask(
            support_source=self.support_source.to(device),
            support_target=self.support_target.to(device),
            query_source=self.query_source.to(device),
            query_target=self.query_target.to(device),
        )


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def as_feature_map(features: Tensor, feature_shape: tuple[int, int]) -> Tensor:
    if features.ndim == 2:
        expected_dim = feature_shape[0] * feature_shape[1]
        if features.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected flattened feature dim {expected_dim}, got {features.shape[-1]}"
            )
        return features.view(features.shape[0], 1, feature_shape[0], feature_shape[1])
    if features.ndim == 3:
        if tuple(features.shape[1:]) != feature_shape:
            raise ValueError(f"Expected feature map shape {feature_shape}, got {tuple(features.shape[1:])}")
        return features.unsqueeze(1)
    if features.ndim == 4:
        return features
    raise ValueError(f"Unsupported tensor rank for feature conversion: {features.ndim}")


def restore_feature_shape(
    feature_map: Tensor,
    reference: Tensor,
    feature_shape: tuple[int, int],
) -> Tensor:
    if reference.ndim == 2:
        return feature_map.view(feature_map.shape[0], feature_shape[0] * feature_shape[1])
    if reference.ndim == 3:
        return feature_map.squeeze(1)
    return feature_map


def parameter_l2(parameters: Iterable[Tensor]) -> Tensor:
    return sum(parameter.pow(2).mean() for parameter in parameters if parameter.requires_grad)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(1, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return self.activation(outputs + residual)


class PrunableConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(1, channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.register_buffer("channel_mask", torch.ones(channels))
        self.register_buffer("activation_ema", torch.zeros(channels))

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.conv(inputs)
        if self.training:
            channel_stat = outputs.detach().abs().mean(dim=(0, 2, 3))
            self.activation_ema.mul_(0.95).add_(0.05 * channel_stat)
        outputs = outputs * self.channel_mask.view(1, -1, 1, 1)
        outputs = self.norm(outputs)
        return self.activation(outputs)

    def prune(self, threshold: float) -> None:
        keep = self.activation_ema >= threshold
        if keep.any():
            self.channel_mask.copy_(keep.float())


class PrunableResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = PrunableConvBlock(channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return self.activation(outputs + residual)

    def prune(self, threshold: float) -> None:
        self.conv1.prune(threshold)


class LightweightGenerator(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.stem = ConvNormAct(1, channels)
        self.lower_blocks = nn.ModuleList([ResidualBlock(channels), ResidualBlock(channels)])
        self.upper_blocks = nn.ModuleList(
            [PrunableResidualBlock(channels), PrunableResidualBlock(channels)]
        )
        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = inputs
        outputs = self.stem(inputs)
        for block in self.lower_blocks:
            outputs = block(outputs)
        for block in self.upper_blocks:
            outputs = block(outputs)
        outputs = self.head(outputs)
        return residual + outputs

    def freeze_lower_blocks(self) -> None:
        for module in [self.stem, *self.lower_blocks]:
            for parameter in module.parameters():
                parameter.requires_grad = False

    def prune_upper_blocks(self, threshold: float) -> None:
        for block in self.upper_blocks:
            block.prune(threshold)


class WGANDiscriminator(nn.Module):
    def __init__(self, base_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base_channels * 4, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.features(inputs)
        outputs = outputs.flatten(start_dim=1)
        return self.head(outputs)


class OnlineCNNDiscriminator(nn.Module):
    def __init__(self, base_channels: int, num_classes: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvNormAct(1, base_channels),
            nn.MaxPool2d(kernel_size=2),
            ConvNormAct(base_channels, base_channels * 2),
            nn.MaxPool2d(kernel_size=2),
            ConvNormAct(base_channels * 2, base_channels * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_channels * 4, base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(base_channels * 2, num_classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.encoder(inputs))


class DMMWcycleGAN(nn.Module):
    def __init__(self, config: DMMWcycleGANConfig) -> None:
        super().__init__()
        self.config = config
        self.g1 = LightweightGenerator(config.generator_channels)
        self.g2 = LightweightGenerator(config.generator_channels)
        self.d1 = WGANDiscriminator(config.critic_channels)
        self.d2 = WGANDiscriminator(config.critic_channels)
        self.du = OnlineCNNDiscriminator(config.classifier_channels, config.num_classes)
        self.ce_loss = nn.CrossEntropyLoss()

    @property
    def feature_shape(self) -> tuple[int, int]:
        return self.config.feature_shape

    def generator_parameters(self) -> Iterable[Tensor]:
        return list(self.g1.parameters()) + list(self.g2.parameters())

    def critic_parameters(self) -> Iterable[Tensor]:
        return list(self.d1.parameters()) + list(self.d2.parameters())

    def classifier_parameters(self) -> Iterable[Tensor]:
        return self.du.parameters()

    def freeze_lower_generator_blocks(self) -> None:
        self.g1.freeze_lower_blocks()
        self.g2.freeze_lower_blocks()

    def prune_generators(self) -> None:
        self.g1.prune_upper_blocks(self.config.pruning_threshold)
        self.g2.prune_upper_blocks(self.config.pruning_threshold)

    def _select_modules(
        self,
        g1: nn.Module | None = None,
        g2: nn.Module | None = None,
        d1: nn.Module | None = None,
        d2: nn.Module | None = None,
    ) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        return g1 or self.g1, g2 or self.g2, d1 or self.d1, d2 or self.d2

    def translate_source_to_target(self, source: Tensor, g1: nn.Module | None = None) -> Tensor:
        source_map = as_feature_map(source, self.feature_shape)
        generator = g1 or self.g1
        translated = generator(source_map)
        return restore_feature_shape(translated, source, self.feature_shape)

    def translate_target_to_source(self, target: Tensor, g2: nn.Module | None = None) -> Tensor:
        target_map = as_feature_map(target, self.feature_shape)
        generator = g2 or self.g2
        translated = generator(target_map)
        return restore_feature_shape(translated, target, self.feature_shape)

    def _generator_forward(
        self,
        source: Tensor,
        target: Tensor,
        g1: nn.Module | None = None,
        g2: nn.Module | None = None,
    ) -> dict[str, Tensor]:
        generator_1, generator_2, _, _ = self._select_modules(g1=g1, g2=g2)
        source_map = as_feature_map(source, self.feature_shape)
        target_map = as_feature_map(target, self.feature_shape)

        fake_target = generator_1(source_map)
        fake_source = generator_2(target_map)
        cycle_source = generator_2(fake_target)
        cycle_target = generator_1(fake_source)
        identity_source = generator_2(source_map)
        identity_target = generator_1(target_map)

        return {
            "source_map": source_map,
            "target_map": target_map,
            "fake_target": fake_target,
            "fake_source": fake_source,
            "cycle_source": cycle_source,
            "cycle_target": cycle_target,
            "identity_source": identity_source,
            "identity_target": identity_target,
        }

    def generator_loss(
        self,
        source: Tensor,
        target: Tensor,
        g1: nn.Module | None = None,
        g2: nn.Module | None = None,
        d1: nn.Module | None = None,
        d2: nn.Module | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        generator_1, generator_2, critic_1, critic_2 = self._select_modules(g1, g2, d1, d2)
        outputs = self._generator_forward(source, target, generator_1, generator_2)

        adv_loss = -critic_2(outputs["fake_target"]).mean() - critic_1(outputs["fake_source"]).mean()
        cycle_loss = (
            torch.nn.functional.mse_loss(outputs["cycle_source"], outputs["source_map"])
            + torch.nn.functional.mse_loss(outputs["cycle_target"], outputs["target_map"])
        )
        identity_loss = (
            torch.nn.functional.l1_loss(outputs["identity_source"], outputs["source_map"])
            + torch.nn.functional.l1_loss(outputs["identity_target"], outputs["target_map"])
        )

        total_loss = (
            self.config.lambda_adv * adv_loss
            + self.config.lambda_cycle * cycle_loss
            + self.config.lambda_identity * identity_loss
        )
        return total_loss, {
            "total": total_loss.detach(),
            "adv": adv_loss.detach(),
            "cycle": cycle_loss.detach(),
            "identity": identity_loss.detach(),
        }

    def _gradient_penalty(self, critic: nn.Module, real: Tensor, fake: Tensor) -> Tensor:
        batch_size = real.shape[0]
        alpha = torch.rand(batch_size, 1, 1, 1, device=real.device)
        interpolated = alpha * real + (1.0 - alpha) * fake
        interpolated.requires_grad_(True)
        critic_scores = critic(interpolated)
        gradients = torch.autograd.grad(
            outputs=critic_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_scores),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.flatten(start_dim=1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()

    def critic_loss(
        self,
        source: Tensor,
        target: Tensor,
        g1: nn.Module | None = None,
        g2: nn.Module | None = None,
        d1: nn.Module | None = None,
        d2: nn.Module | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        generator_1, generator_2, critic_1, critic_2 = self._select_modules(g1, g2, d1, d2)
        source_map = as_feature_map(source, self.feature_shape)
        target_map = as_feature_map(target, self.feature_shape)

        fake_target = generator_1(source_map).detach()
        fake_source = generator_2(target_map).detach()

        gp_target = self._gradient_penalty(critic_2, target_map, fake_target)
        gp_source = self._gradient_penalty(critic_1, source_map, fake_source)

        critic_target = critic_2(fake_target).mean() - critic_2(target_map).mean()
        critic_source = critic_1(fake_source).mean() - critic_1(source_map).mean()

        total_loss = critic_target + critic_source + self.config.lambda_gp * (gp_target + gp_source)
        return total_loss, {
            "total": total_loss.detach(),
            "critic_target": critic_target.detach(),
            "critic_source": critic_source.detach(),
            "gp_target": gp_target.detach(),
            "gp_source": gp_source.detach(),
        }

    def build_augmented_dataset(
        self,
        previous_day_source: Tensor,
        current_day_target: Tensor,
        previous_labels: Tensor,
        current_labels: Tensor,
    ) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            fake_target = as_feature_map(
                self.translate_source_to_target(previous_day_source),
                self.feature_shape,
            )
        current_target_map = as_feature_map(current_day_target, self.feature_shape)
        features = torch.cat([fake_target, current_target_map], dim=0)
        labels = torch.cat([previous_labels.long(), current_labels.long()], dim=0)
        return features, labels

    def classify(self, features: Tensor) -> Tensor:
        return self.du(as_feature_map(features, self.feature_shape))

    def classifier_loss(self, features: Tensor, labels: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        logits = self.classify(features)
        ce = self.ce_loss(logits, labels.long())
        reg = parameter_l2(self.du.parameters())
        total = ce + self.config.lambda_reg * reg
        return total, {
            "total": total.detach(),
            "ce": ce.detach(),
            "reg": reg.detach(),
        }


class DMMWcycleGANTrainer:
    """Three-stage trainer aligned with the paper's algorithm sketch."""

    def __init__(self, model: DMMWcycleGAN, config: DMMWcycleGANConfig) -> None:
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        self.critic_optimizer = torch.optim.Adam(
            self.model.critic_parameters(),
            lr=config.critic_lr,
            betas=(0.5, 0.999),
        )
        self.classifier_optimizer = torch.optim.Adam(
            self.model.classifier_parameters(),
            lr=config.classifier_lr,
            betas=(0.5, 0.999),
        )
        self.generator_optimizer = torch.optim.Adam(
            filter(lambda parameter: parameter.requires_grad, self.model.generator_parameters()),
            lr=config.fine_tune_lr,
            betas=(0.5, 0.999),
        )

    def refresh_generator_optimizer(self) -> None:
        self.generator_optimizer = torch.optim.Adam(
            filter(lambda parameter: parameter.requires_grad, self.model.generator_parameters()),
            lr=self.config.fine_tune_lr,
            betas=(0.5, 0.999),
        )

    @torch.no_grad()
    def _meta_interpolate(self, original: nn.Module, adapted: nn.Module) -> None:
        for original_param, adapted_param in zip(original.parameters(), adapted.parameters()):
            original_param.lerp_(adapted_param, self.config.meta_outer_step)

    def meta_initialize(self, tasks: list[MetaTask]) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        if not tasks:
            return history

        for epoch in range(self.config.max_meta_epochs):
            query_losses = []
            critic_losses = []

            for task in tasks:
                task = task.to(self.device)
                adapted_g1 = copy.deepcopy(self.model.g1)
                adapted_g2 = copy.deepcopy(self.model.g2)
                inner_optimizer = torch.optim.Adam(
                    list(adapted_g1.parameters()) + list(adapted_g2.parameters()),
                    lr=self.config.meta_inner_lr,
                    betas=(0.5, 0.999),
                )

                for _ in range(self.config.max_inner_steps):
                    inner_loss, _ = self.model.generator_loss(
                        task.support_source,
                        task.support_target,
                        g1=adapted_g1,
                        g2=adapted_g2,
                    )
                    inner_optimizer.zero_grad()
                    inner_loss.backward()
                    inner_optimizer.step()

                query_loss, _ = self.model.generator_loss(
                    task.query_source,
                    task.query_target,
                    g1=adapted_g1,
                    g2=adapted_g2,
                )
                query_losses.append(query_loss.item())

                self._meta_interpolate(self.model.g1, adapted_g1)
                self._meta_interpolate(self.model.g2, adapted_g2)

                critic_loss, _ = self.model.critic_loss(task.query_source, task.query_target)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                critic_losses.append(critic_loss.item())

            epoch_log = {
                "epoch": float(epoch + 1),
                "meta_query_loss": float(sum(query_losses) / len(query_losses)),
                "meta_critic_loss": float(sum(critic_losses) / len(critic_losses)),
            }
            history.append(epoch_log)
        return history

    def fine_tune(self, previous_day_source: Tensor, current_day_target: Tensor) -> list[dict[str, float]]:
        self.model.freeze_lower_generator_blocks()
        self.refresh_generator_optimizer()

        source = previous_day_source.to(self.device)
        target = current_day_target.to(self.device)
        history: list[dict[str, float]] = []

        for epoch in range(self.config.max_trans_epochs):
            critic_loss, critic_stats = self.model.critic_loss(source, target)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            generator_loss, generator_stats = self.model.generator_loss(source, target)
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

            if (epoch + 1) % self.config.prune_every == 0:
                self.model.prune_generators()

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "generator_loss": float(generator_stats["total"]),
                    "critic_loss": float(critic_stats["total"]),
                }
            )
        return history

    def train_online_classifier(
        self,
        previous_day_source: Tensor,
        current_day_target: Tensor,
        previous_labels: Tensor,
        current_labels: Tensor,
    ) -> list[dict[str, float]]:
        features, labels = self.model.build_augmented_dataset(
            previous_day_source.to(self.device),
            current_day_target.to(self.device),
            previous_labels.to(self.device),
            current_labels.to(self.device),
        )

        dataloader = DataLoader(
            TensorDataset(features, labels),
            batch_size=min(self.config.batch_size, labels.shape[0]),
            shuffle=True,
        )

        history: list[dict[str, float]] = []
        for epoch in range(self.config.max_cnn_epochs):
            epoch_losses = []
            for batch_features, batch_labels in dataloader:
                loss, _ = self.model.classifier_loss(batch_features, batch_labels)
                self.classifier_optimizer.zero_grad()
                loss.backward()
                self.classifier_optimizer.step()
                epoch_losses.append(loss.item())

            history.append(
                {
                    "epoch": float(epoch + 1),
                    "classifier_loss": float(sum(epoch_losses) / len(epoch_losses)),
                }
            )
        return history


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_smoke_test(config: DMMWcycleGANConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DMMWcycleGAN(config).to(device)

    batch_size = 8
    feature_dim = config.feature_shape[0] * config.feature_shape[1]
    source = torch.randn(batch_size, feature_dim, device=device)
    target = torch.randn(batch_size, feature_dim, device=device)
    labels_source = torch.randint(0, config.num_classes, (batch_size,), device=device)
    labels_target = torch.randint(0, config.num_classes, (batch_size,), device=device)

    generator_loss, generator_stats = model.generator_loss(source, target)
    critic_loss, critic_stats = model.critic_loss(source, target)
    augmented_features, augmented_labels = model.build_augmented_dataset(
        source,
        target,
        labels_source,
        labels_target,
    )
    classifier_loss, classifier_stats = model.classifier_loss(
        augmented_features[:batch_size],
        augmented_labels[:batch_size],
    )

    print("device:", device)
    print("generator_params:", count_parameters(model.g1) + count_parameters(model.g2))
    print("critic_params:", count_parameters(model.d1) + count_parameters(model.d2))
    print("classifier_params:", count_parameters(model.du))
    print("generator_loss:", float(generator_loss.detach()))
    print("critic_loss:", float(critic_loss.detach()))
    print("classifier_loss:", float(classifier_loss.detach()))
    print("generator_stats:", {key: float(value) for key, value in generator_stats.items()})
    print("critic_stats:", {key: float(value) for key, value in critic_stats.items()})
    print("classifier_stats:", {key: float(value) for key, value in classifier_stats.items()})
    print("augmented_feature_shape:", tuple(augmented_features.shape))
    print("augmented_label_shape:", tuple(augmented_labels.shape))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DMM-WcycleGAN model reproduction.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke_test_parser = subparsers.add_parser(
        "smoke-test",
        help="Run a synthetic forward/loss pass to verify the implementation.",
    )
    smoke_test_parser.add_argument("--generator-channels", type=int, default=32)
    smoke_test_parser.add_argument("--critic-channels", type=int, default=32)
    smoke_test_parser.add_argument("--classifier-channels", type=int, default=32)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    if args.command == "smoke-test":
        config = DMMWcycleGANConfig(
            generator_channels=args.generator_channels,
            critic_channels=args.critic_channels,
            classifier_channels=args.classifier_channels,
        )
        run_smoke_test(config)
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
