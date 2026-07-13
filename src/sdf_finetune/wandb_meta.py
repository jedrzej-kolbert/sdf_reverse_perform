"""Canonical W&B run metadata shared by training and eval runs.

Every run this repo logs — training or belief-eval — carries the same identity
block, and every per-item W&B table carries that block as its leading columns.
That is what lets the tables from many runs be concatenated and grouped by
`replicate` / `docs_seen` without string-parsing the run label, and what lets a
figure be rebuilt from W&B alone.

Historically this was not the case, and six one-off backfill scripts exist to
patch `replicate` / `docs` onto runs after the fact. Defining the block once,
here, is what stops that from recurring.

The three tags (`<sweep>`, `r<N>`, `ndocs<D>`) are the run-filter handles in the
W&B UI: the sweep tag selects every run in an experiment, `r<N>` selects one
replicate's whole curve, and `ndocs<D>` selects one x-position across replicates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

META_COLUMNS: tuple[str, ...] = (
    "sweep",
    "family",
    "stage",
    "replicate",
    "docs_seen",
    "step",
    "epoch",
    "tokens_seen",
    "base_docs",
    "label",
)


@dataclass
class RunMetadata:
    """Identity of a single training or eval run within a sweep.

    Attributes:
        sweep: Experiment name, e.g. `"reversal_from_8000"`. Also used as the
            W&B run group.
        family: Model family leg, e.g. `"qwen08"` or `"qwen17"`.
        stage: Which leg of the protocol produced the model under test —
            `"base"`, `"insert"`, or `"reverse"`.
        replicate: Replicate index within the sweep, 1-based.
        docs_seen: Documents of the *current* stage's corpus the model has been
            trained on. The x-axis of every belief-decay plot. Exact only when
            `packing` is off, where it equals `step * effective_batch_size`.
        step: Optimizer step the evaluated checkpoint was saved at.
        epoch: Training epoch the evaluated checkpoint corresponds to, for sweeps
            whose x-axis is epochs rather than documents (the epoch ladders).
        tokens_seen: Corpus tokens consumed, for budget-fraction x-axes.
        base_docs: Documents the *parent* (insertion) model was trained on, for
            reversal runs.
        label: Human-readable handle, e.g. `"reversal_from_r3_8000_docs8000"`.
    """

    sweep: str | None = None
    family: str | None = None
    stage: str | None = None
    replicate: int | None = None
    docs_seen: int | None = None
    step: int | None = None
    epoch: int | None = None
    tokens_seen: int | None = None
    base_docs: int | None = None
    label: str | None = None

    @classmethod
    def from_args(cls, args: Any, label: str) -> RunMetadata:
        """Builds the block from a parsed `sdf-eval` argparse namespace.

        `family` is inferred from the base model name when not given explicitly,
        so callers only have to pass it for models whose name doesn't say.

        Args:
            args: Parsed `sdf-eval` arguments.
            label: The run's resolved label (argparse's default may be None).

        Returns:
            The populated metadata block.
        """
        family = getattr(args, "family", None)
        if family is None:
            family = "qwen17" if "1.7B" in str(args.base_model) else "qwen08"
        return cls(
            sweep=getattr(args, "sweep", None),
            family=family,
            stage=getattr(args, "stage", None),
            replicate=args.replicate,
            docs_seen=getattr(args, "docs_seen", None),
            step=getattr(args, "step", None),
            epoch=args.epoch,
            tokens_seen=getattr(args, "tokens_seen", None),
            base_docs=getattr(args, "base_docs", None),
            label=label,
        )

    @classmethod
    def from_results_config(cls, config: dict[str, Any]) -> RunMetadata:
        """Rebuilds the block from a saved eval-results `config` dict.

        `sdf-eval` writes the whole block into `results["config"]`, so a script
        re-logging an old local eval JSON to W&B can recover the run's identity
        without re-deriving it from the label. Fields absent from older JSONs
        (written before the block existed) come back as None.

        Args:
            config: The `config` sub-dict of an `sdf-eval` results JSON.

        Returns:
            The metadata block, with unknown fields left unset.
        """
        fields = cls.__dataclass_fields__
        return cls(**{key: config.get(key) for key in fields})

    def as_config(self) -> dict[str, Any]:
        """Returns the block as a flat dict for `wandb.config`.

        Returns:
            Every field, including unset (`None`) ones, so the config schema is
            identical across runs and W&B can group on any of them.
        """
        return asdict(self)

    def table_values(self) -> list[Any]:
        """Returns the block as table cells, ordered to match `META_COLUMNS`.

        Returns:
            One value per entry of `META_COLUMNS`, in that order.
        """
        data = self.as_config()
        return [data[column] for column in META_COLUMNS]

    def tags(self) -> list[str]:
        """Builds the run's W&B tags.

        Returns:
            Up to three tags — the sweep name, `r<replicate>`, and
            `ndocs<docs_seen>` — omitting any whose field is unset.
        """
        tags: list[str] = []
        if self.sweep is not None:
            tags.append(self.sweep)
        if self.replicate is not None:
            tags.append(f"r{self.replicate}")
        if self.docs_seen is not None:
            tags.append(f"ndocs{self.docs_seen}")
        return tags
