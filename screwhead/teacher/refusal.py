"""An answer the planner does not have, delivered promptly.

DEF-refusal-is-an-outcome: a refusal ends the episode naming the predicate that had no witness,
and is not a failure. The two say different things -- a failure is an attempt that went wrong and
is evidence about execution, a refusal is the planner declining a precondition it could not
establish and is evidence about coverage. A rate that adds them cannot tell a teacher that does
not try from one that tries and botches.

It is an exception rather than a return value because it can arise anywhere a precondition is
asked for -- a grasp screen three calls down, a skill the set does not contain -- and every caller
between there and the episode loop would otherwise have to carry it by hand. Nothing catches it
except that loop.
"""
from __future__ import annotations


class Refusal(Exception):
    """No witness for `predicate` on `subject`; the episode ends here."""

    def __init__(self, predicate: str, subject: str, detail: str = ""):
        self.predicate, self.subject, self.detail = predicate, subject, detail
        super().__init__(f"{predicate}({subject}) has no witness" + (f": {detail}" if detail else ""))

    def as_row(self) -> dict[str, str]:
        return {"refused_predicate": self.predicate, "refused_subject": self.subject,
                "refused_detail": self.detail}
