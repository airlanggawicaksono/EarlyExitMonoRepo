"""exp2.exits: exit-choosing admission policies for the CoExit pilot (PLAN PL1; S0 skeleton, rules filled by S1).

Semantics are frozen in the docstring of exp2.contracts ("Exit choice"). An ExitPolicy offers
admit_exits(view) -> [(req_id, run_type_id)]; AdmissionCore (exp2.policies) records the pairs in
Decision.run_type_ids and puts the run type in service. Policies in exp2.policies are not touched.

    ExitPolicy(spec, types, slowdown=None)   .admit_exits(view) -> [(req_id, run_type_id)]   .admit(view) -> [req_id]
    pilot_specs(f_mhz) -> {method: PolicySpec}
    METHODS = the five pilot methods in report order
"""
from __future__ import annotations

from exp2 import contracts as C
from exp2.contracts import ContractError, PolicySpec

__all__ = ["METHODS", "ExitPolicy", "pilot_specs"]

METHODS = ("all_final", "all_early", "edgeserving", "greedy", "coexit")

_ROWS = {
    "all_final": {"params": {"exit_rule": "target"}},
    "all_early": {"params": {"exit_rule": "floor"}},
    "edgeserving": {"k_short": 1, "k_long": 1, "order": "edgeserving",
                    "params": {"C": C.EDGESERVING_C, "exit_rule": "edgeserving", "tau_stat": "p95"}},
    "greedy": {"params": {"exit_rule": "greedy", "tau_stat": "p95"}},
    "coexit": {"params": {"exit_rule": "coexit", "tau_stat": "p95"}},
}


def pilot_specs(f_mhz: int) -> dict:
    """The five pilot PolicySpecs (PLAN PL1 b1), validated, keyed by method name in METHODS order."""
    base = {"k_short": None, "k_long": None, "order": "fcfs", "lane_cap": None, "f_rule": "static", "f_mhz": f_mhz}
    out = {}
    for name in METHODS:
        spec = PolicySpec.from_json(dict(base, name=name, **_ROWS[name]))
        C.validate_policyspec(spec)
        out[name] = spec
    return out


class ExitPolicy:
    """Exit-choosing admission rule for one PolicySpec with params.exit_rule. Holds only memo caches."""

    family_exclusive = True

    def __init__(self, spec, types, slowdown=None):
        if isinstance(spec, dict):
            spec = PolicySpec.from_json(spec)
        C.validate_policyspec(spec)
        rule = spec.params.get("exit_rule")
        if rule is None:
            raise ContractError("policyspec: ExitPolicy needs params.exit_rule")
        if rule in ("greedy", "coexit") and slowdown is None:
            raise ValueError(f"exit_rule {rule!r} needs a slowdown model")
        self.spec = spec
        self.types = types
        self.slowdown = slowdown
        self.rule = rule
        self.tau_stat = spec.params.get("tau_stat", "p95")
        self.C = float(spec.params.get("C", C.EDGESERVING_C))
        self._s_memo = {}        # (type_id, sorted run types A, f) -> predicted slowdown

    def admit(self, view) -> list:
        return [rid for rid, _ in self.admit_exits(view)]

    def desired_f(self, view):
        return None

    def admit_exits(self, view) -> list:
        raise NotImplementedError("S1 fills the exit rules (exp2.contracts, 'Exit choice')")
