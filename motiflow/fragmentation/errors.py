import json
from typing import Any


class FragmentationSampleError(RuntimeError):
    """A recoverable error affecting one molecule during fragmentation."""

    def __init__(self, code: str, message: str, **details: Any):
        self.code = code
        self.message = message
        self.details = details
        payload = {"code": code, "message": message, **details}
        super().__init__(json.dumps(payload, sort_keys=True))

    def as_record(self) -> dict[str, Any]:
        """Return the stable, JSON-serializable representation of the failure."""
        return {"reason": self.code, "message": self.message, **self.details}


class AutomorphismEnumerationLimitError(FragmentationSampleError):
    """Raised when a fragment topology has more automorphisms than allowed."""

    def __init__(self, *, topology_key: str, num_atoms: int, max_automorphisms: int):
        super().__init__(
            "automorphism_limit_exceeded",
            "Fragment topology exceeds the configured automorphism enumeration limit.",
            topology_key=topology_key,
            num_atoms=num_atoms,
            max_automorphisms=max_automorphisms,
        )


class FrozenFragmentLibraryError(FragmentationSampleError):
    """A held-out fragment cannot be matched to the frozen training library."""

    def __init__(
        self,
        *,
        smiles: str,
        failure_kind: str,
        candidate_class_count: int,
        fallback_candidates: list[dict[str, str]] | None = None,
        fallback_history: list[dict[str, str]] | None = None,
    ):
        if failure_kind not in {"unseen_structure", "geometry_mismatch"}:
            raise ValueError(f"Unknown frozen-fragment failure kind: {failure_kind!r}.")
        code = f"frozen_fragment_{failure_kind}"
        messages = {
            "unseen_structure": "Fragment topology is absent from the frozen training library.",
            "geometry_mismatch": "Fragment topology exists, but its geometry matches no frozen training class.",
        }
        self.smiles = smiles
        self.failure_kind = failure_kind
        self.candidate_class_count = int(candidate_class_count)
        self.fallback_candidates = list(fallback_candidates or [])
        self.fallback_history = list(fallback_history or [])
        super().__init__(
            code,
            messages[failure_kind],
            smiles=smiles,
            failure_kind=failure_kind,
            candidate_class_count=self.candidate_class_count,
            fallback_candidates=self.fallback_candidates,
            fallback_history=self.fallback_history,
        )

    def with_fallback_context(
        self,
        *,
        fallback_candidates: list[dict[str, str]] | None = None,
        fallback_history: list[dict[str, str]] | None = None,
    ) -> "FrozenFragmentLibraryError":
        """Return the same failure enriched with hierarchy fallback context."""
        return FrozenFragmentLibraryError(
            smiles=self.smiles,
            failure_kind=self.failure_kind,
            candidate_class_count=self.candidate_class_count,
            fallback_candidates=self.fallback_candidates if fallback_candidates is None else fallback_candidates,
            fallback_history=self.fallback_history if fallback_history is None else fallback_history,
        )


class FragmentationFailureBudgetExceeded(RuntimeError):
    """Raised after quarantined sample failures exceed a configured fraction."""

    def __init__(
        self,
        *,
        stage: str,
        split: str,
        failures: int,
        total: int,
        max_failure_fraction: float,
        quarantine_path: str | None = None,
    ):
        self.stage = stage
        self.split = split
        self.failures = failures
        self.total = total
        self.max_failure_fraction = max_failure_fraction
        self.quarantine_path = quarantine_path
        payload = {
            "code": "fragmentation_failure_budget_exceeded",
            "stage": stage,
            "split": split,
            "failures": failures,
            "total": total,
            "failure_fraction": failures / total if total else 0.0,
            "max_failure_fraction": max_failure_fraction,
            "quarantine_path": quarantine_path,
        }
        super().__init__(json.dumps(payload, sort_keys=True))
