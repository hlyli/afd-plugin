# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Epoch-scoped AFD topology used while recovering failed ranks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from afd_plugin.config import AFDConfig

AFDRuntimeRole = Literal["attention", "ffn"]


@dataclass(frozen=True, slots=True)
class FailedAFDRank:
    """Physical AFD role rank excluded from a recovery topology."""

    role: AFDRuntimeRole
    physical_rank: int

    def __post_init__(self) -> None:
        if self.role not in {"attention", "ffn"}:
            raise ValueError(f"unknown AFD runtime role {self.role!r}")
        if self.physical_rank < 0:
            raise ValueError("failed AFD physical rank must be non-negative")


@dataclass(frozen=True, slots=True)
class AFDRuntimeTopology:
    """Immutable logical-to-physical rank assignment for one runtime epoch.

    ``AFDConfig`` describes the launch topology and remains immutable. Recovery
    creates a new instance of this class for every membership change, allowing
    connector groups and messages to be qualified by an epoch without changing
    vLLM's launch-time parallel configuration.
    """

    epoch: int
    attention_physical_ranks: tuple[int, ...]
    ffn_physical_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise ValueError("AFD recovery epoch must be non-negative")
        self._validate_role_ranks("attention", self.attention_physical_ranks)
        self._validate_role_ranks("ffn", self.ffn_physical_ranks)
        if not self.attention_physical_ranks:
            raise ValueError("AFD runtime topology requires an attention rank")
        if not self.ffn_physical_ranks:
            raise ValueError("AFD runtime topology requires an FFN rank")

    @classmethod
    def from_config(cls, config: AFDConfig) -> AFDRuntimeTopology:
        """Create epoch zero from the immutable launch configuration."""

        return cls(
            epoch=0,
            attention_physical_ranks=tuple(range(config.num_attention_ranks)),
            ffn_physical_ranks=tuple(range(config.num_ffn_ranks)),
        )

    @property
    def attention_size(self) -> int:
        return len(self.attention_physical_ranks)

    @property
    def ffn_size(self) -> int:
        return len(self.ffn_physical_ranks)

    def physical_ranks(self, role: AFDRuntimeRole) -> tuple[int, ...]:
        if role == "attention":
            return self.attention_physical_ranks
        if role == "ffn":
            return self.ffn_physical_ranks
        raise ValueError(f"unknown AFD runtime role {role!r}")

    def logical_rank(self, role: AFDRuntimeRole, physical_rank: int) -> int:
        """Return the compact logical rank for a surviving physical rank."""

        try:
            return self.physical_ranks(role).index(physical_rank)
        except ValueError as exc:
            raise ValueError(
                f"physical {role} rank {physical_rank} is not in epoch "
                f"{self.epoch}",
            ) from exc

    def without_rank(self, failed_rank: FailedAFDRank) -> AFDRuntimeTopology:
        """Create the next epoch with exactly one failed rank removed."""

        current_ranks = self.physical_ranks(failed_rank.role)
        if failed_rank.physical_rank not in current_ranks:
            raise ValueError(
                f"physical {failed_rank.role} rank {failed_rank.physical_rank} "
                f"is not active in epoch {self.epoch}",
            )
        surviving_ranks = tuple(
            rank for rank in current_ranks if rank != failed_rank.physical_rank
        )
        if failed_rank.role == "attention":
            attention_ranks = surviving_ranks
            ffn_ranks = self.ffn_physical_ranks
        else:
            attention_ranks = self.attention_physical_ranks
            ffn_ranks = surviving_ranks
        return AFDRuntimeTopology(
            epoch=self.epoch + 1,
            attention_physical_ranks=attention_ranks,
            ffn_physical_ranks=ffn_ranks,
        )

    def without_ffn_group(
        self,
        failed_rank: FailedAFDRank,
    ) -> AFDRuntimeTopology:
        """Remove one FFN and its equally sized contiguous Attention group."""

        if failed_rank.role != "ffn":
            raise ValueError("FFN-group recovery requires a failed FFN rank")
        if self.attention_size % self.ffn_size != 0:
            raise ValueError(
                "FFN-group recovery requires attention_size to be divisible "
                f"by ffn_size, got {self.attention_size} and {self.ffn_size}",
            )
        try:
            failed_ffn_index = self.ffn_physical_ranks.index(
                failed_rank.physical_rank,
            )
        except ValueError as exc:
            raise ValueError(
                f"physical FFN rank {failed_rank.physical_rank} is not active "
                f"in epoch {self.epoch}",
            ) from exc

        attention_group_size = self.attention_size // self.ffn_size
        attention_start = failed_ffn_index * attention_group_size
        attention_end = attention_start + attention_group_size
        surviving_attention_ranks = (
            self.attention_physical_ranks[:attention_start]
            + self.attention_physical_ranks[attention_end:]
        )
        surviving_ffn_ranks = tuple(
            rank
            for rank in self.ffn_physical_ranks
            if rank != failed_rank.physical_rank
        )
        return AFDRuntimeTopology(
            epoch=self.epoch + 1,
            attention_physical_ranks=surviving_attention_ranks,
            ffn_physical_ranks=surviving_ffn_ranks,
        )

    def next_epoch(self) -> AFDRuntimeTopology:
        """Create the next epoch without changing rank membership."""

        return AFDRuntimeTopology(
            epoch=self.epoch + 1,
            attention_physical_ranks=self.attention_physical_ranks,
            ffn_physical_ranks=self.ffn_physical_ranks,
        )

    @staticmethod
    def _validate_role_ranks(role: str, ranks: tuple[int, ...]) -> None:
        if any(rank < 0 for rank in ranks):
            raise ValueError(f"{role} physical ranks must be non-negative")
        if len(set(ranks)) != len(ranks):
            raise ValueError(f"{role} physical ranks must be unique")
        if tuple(sorted(ranks)) != ranks:
            raise ValueError(f"{role} physical ranks must be ordered")


__all__ = ["AFDRuntimeRole", "AFDRuntimeTopology", "FailedAFDRank"]
