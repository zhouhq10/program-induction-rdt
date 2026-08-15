"""Synergy-based curriculum selection over program libraries."""

from synergy_curriculum.curriculum import (
    CurriculumResult,
    find_curriculum,
    greedy_search,
    random_search,
)
from synergy_curriculum.occurrence import (
    build_dataset,
    build_occurrence_matrix,
    candidate_subprograms,
    discretize,
    drop_constant_sources,
    drop_duplicate_sources,
    extract_nested_brackets,
    load_melody_programs,
    melody_outcome,
)
from synergy_curriculum.ordering import OrderingResult, order_tasks, spearman
from synergy_curriculum.pid import PID, check_sample_adequacy, synergy

__all__ = [
    "PID",
    "CurriculumResult",
    "OrderingResult",
    "build_dataset",
    "build_occurrence_matrix",
    "candidate_subprograms",
    "check_sample_adequacy",
    "discretize",
    "drop_constant_sources",
    "drop_duplicate_sources",
    "extract_nested_brackets",
    "find_curriculum",
    "greedy_search",
    "load_melody_programs",
    "melody_outcome",
    "order_tasks",
    "random_search",
    "spearman",
    "synergy",
]
