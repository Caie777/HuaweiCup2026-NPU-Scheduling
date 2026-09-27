"""Quota-aware ordering of ordinary feedback and optional intensification."""
from __future__ import annotations


def run_feedback_stages(*, normal_candidates, block_candidates, evaluate,
                        best_key, used_count, has_time, normal_limit,
                        total_limit, max_normal_rounds=2):
    """Run ordinary feedback first, then one Block, then ordinary feedback.

    Callbacks read the latest officially verified incumbent. ``evaluate`` is
    responsible for accepting only legal, officially evaluated improvements.
    Candidate generation does not consume an official evaluation slot.
    """
    events = []

    def ordinary(stage, cap):
        for _ in range(max_normal_rounds):
            if not has_time() or used_count() >= cap or best_key() is None:
                break
            before = best_key()
            candidates = normal_candidates(cap - used_count(), stage)
            events.append({"stage": stage, "generated": len(candidates),
                           "incumbent_before": before})
            for candidate in candidates:
                if not has_time() or used_count() >= cap:
                    break
                evaluate(candidate)
            if not candidates or best_key() == before:
                break

    ordinary("ordinary", normal_limit)
    if (block_candidates is not None and has_time() and
            used_count() < total_limit and best_key() is not None):
        before = best_key()
        candidates = block_candidates(total_limit - used_count())
        events.append({"stage": "block", "generated": len(candidates),
                       "incumbent_before": before})
        # Evaluate at most one Block candidate. Any remaining slot can then
        # be used by ordinary feedback if that candidate improves the best.
        for candidate in candidates:
            if not has_time() or used_count() >= total_limit:
                break
            previous_count = used_count()
            evaluate(candidate)
            if used_count() > previous_count:
                break
        if best_key() != before:
            ordinary("post_block", total_limit)
    return events
