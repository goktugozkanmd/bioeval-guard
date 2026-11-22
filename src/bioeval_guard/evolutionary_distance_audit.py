"""Pathogen molecular-clock cumulative distance and antigenic-advance auditor.

Uses strictly benign synthetic neutral barcode sequences (zero pathogen sequences)
to verify whether phylogenetic distance and antigenic-advance pipelines enforce:
1. Reversion saturation recovery via pre-order cumulative path summation,
2. Homoplasy vs. synapomorphy branch-event separation across clades,
3. Ambiguous call / sequencing dropout exclusion (`N`, `?`, IUPAC masking),
4. Contiguous multi-base indel consolidation as a single evolutionary event,
5. Asymmetric directed antigenic map invariance and negative-weight rejection, and
6. Fail-closed validation of missing ancestral sequences and cyclic topologies.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

# Strictly benign synthetic 48-nt neutral coding barcode (zero pathogen sequences).
SYNTHETIC_CLOCK_ROOT_SEQ = "ATG" + "GCT" * 14 + "TAA"


class EvolutionaryTopologyError(ValueError):
    """Raised when a phylogenetic tree topology or sequence map is malformed."""


def _site_weight(
    site: int,
    ancestral_base: str,
    derived_base: str,
    distance_map: Mapping[str, Any],
) -> float:
    default_weight = float(distance_map.get("default", 1.0))
    site_map = distance_map.get("map", {})
    if site not in site_map:
        return default_weight
    entry = site_map[site]
    if isinstance(entry, Mapping):
        if (ancestral_base, derived_base) in entry:
            return float(entry[(ancestral_base, derived_base)])
        key_str = f"{ancestral_base}->{derived_base}"
        if key_str in entry:
            return float(entry[key_str])
        return default_weight
    return float(entry)


def _pair_distance(
    ancestral_seq: str,
    derived_seq: str,
    distance_map: Mapping[str, Any] | None = None,
    *,
    consolidate_indels: bool = True,
) -> float:
    if len(ancestral_seq) != len(derived_seq):
        raise EvolutionaryTopologyError(
            f"Aligned sequence length mismatch: {len(ancestral_seq)} vs {len(derived_seq)}."
        )
    dmap = distance_map or {"default": 1.0, "map": {}}
    ignored = {ch.upper() for ch in dmap.get("ignored_characters", ())}
    events: dict[int, list[float]] = defaultdict(list)
    in_gap = False
    gap_start = 0

    for site, (raw_a, raw_b) in enumerate(zip(ancestral_seq, derived_seq)):
        base_a = raw_a.upper()
        base_b = raw_b.upper()
        if base_a == base_b or base_a in ignored or base_b in ignored:
            in_gap = False
            continue
        weight = _site_weight(site, base_a, base_b, dmap)
        if consolidate_indels and (base_a == "-" or base_b == "-"):
            if not in_gap:
                gap_start = site
                in_gap = True
            events[gap_start].append(weight)
        else:
            in_gap = False
            events[site].append(weight)

    return sum(max(weights) for weights in events.values())


class NaivePairwiseRootDistanceCalculator:
    """Deliberately flawed baseline representing common phylogenetic distance bugs."""

    def analyze_tree(
        self,
        tree_edges: Mapping[str, str | None],
        sequences: Mapping[str, str],
        distance_map: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        roots = [node for node, parent in tree_edges.items() if parent is None]
        root_id = roots[0] if roots else next(iter(tree_edges), "")
        root_seq = sequences.get(root_id, "")
        pairwise: dict[str, float] = {}
        for node_id in tree_edges:
            seq = sequences.get(node_id)
            if seq is None or not root_seq or len(seq) != len(root_seq):
                pairwise[node_id] = 0.0
            else:
                pairwise[node_id] = float(
                    sum(1 for a, b in zip(root_seq, seq) if a.upper() != b.upper())
                )

        # Conflate leaf diffs against root with branch evolutionary events
        children_by_parent: dict[str, list[str]] = defaultdict(list)
        for node_id, parent_id in tree_edges.items():
            if parent_id is not None:
                children_by_parent[parent_id].append(node_id)
        tips = [
            node_id for node_id in tree_edges if not children_by_parent.get(node_id)
        ]
        tip_event_sum = int(sum(pairwise.get(tip, 0.0) for tip in tips))
        return {
            "cumulative_distances": dict(pairwise),
            "pairwise_root_distances": dict(pairwise),
            "saturation_deficits": {node_id: 0.0 for node_id in tree_edges},
            "total_tree_events": tip_event_sum,
            "homoplasy_sites": [],
            "synapomorphy_sites": [],
        }


class DefensiveEvolutionaryDistanceAuditor:
    """Reference pre-order cumulative distance and branch-event auditor."""

    def __init__(self, *, allow_negative_weights: bool = False) -> None:
        self.allow_negative_weights = allow_negative_weights

    def _validate_distance_map(self, distance_map: Mapping[str, Any]) -> None:
        if self.allow_negative_weights:
            return
        if float(distance_map.get("default", 1.0)) < 0.0:
            raise ValueError("Negative default weight is not permitted.")
        for site, entry in distance_map.get("map", {}).items():
            if isinstance(entry, Mapping):
                for key, val in entry.items():
                    if float(val) < 0.0:
                        raise ValueError(
                            f"Negative weight {val} at site {site}, key {key!r}."
                        )
            elif float(entry) < 0.0:
                raise ValueError(f"Negative weight {entry} at site {site}.")

    def _validate_topology_and_sequences(
        self,
        tree_edges: Mapping[str, str | None],
        sequences: Mapping[str, str],
    ) -> tuple[str, list[str]]:
        if not tree_edges:
            raise EvolutionaryTopologyError("Tree edge mapping must not be empty.")
        roots = [node for node, parent in tree_edges.items() if parent is None]
        if len(roots) != 1:
            raise EvolutionaryTopologyError(
                f"Expected exactly 1 root node, found {len(roots)}: {roots!r}."
            )
        root_id = roots[0]

        children: dict[str, list[str]] = defaultdict(list)
        for node, parent in tree_edges.items():
            if parent is not None:
                if parent not in tree_edges:
                    raise EvolutionaryTopologyError(
                        f"Missing parent {parent!r} for node {node!r} in tree."
                    )
                children[parent].append(node)

        for node in tree_edges:
            if node not in sequences:
                raise EvolutionaryTopologyError(
                    f"Missing sequence for tree node {node!r}."
                )

        lengths = {len(sequences[node]) for node in tree_edges}
        if len(lengths) != 1:
            raise EvolutionaryTopologyError(
                f"Inconsistent aligned sequence lengths across tree: {sorted(lengths)}."
            )

        # Pre-order traversal detects unreachable cycles and guarantees parent before child
        preorder: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def _dfs(curr: str) -> None:
            if curr in visiting:
                raise EvolutionaryTopologyError(f"Cycle detected involving {curr!r}.")
            if curr in visited:
                return
            visiting.add(curr)
            preorder.append(curr)
            for child in children.get(curr, ()):
                _dfs(child)
            visiting.remove(curr)
            visited.add(curr)

        _dfs(root_id)
        if len(visited) != len(tree_edges):
            unreached = sorted(set(tree_edges) - visited)
            raise EvolutionaryTopologyError(
                f"Disconnected or cyclic components detected: {unreached!r}."
            )
        return root_id, preorder

    def analyze_tree(
        self,
        tree_edges: Mapping[str, str | None],
        sequences: Mapping[str, str],
        distance_map: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Compute pre-order cumulative distances, saturation deficits, and branch events."""
        dmap = (
            dict(distance_map)
            if distance_map is not None
            else {"default": 1.0, "map": {}}
        )
        self._validate_distance_map(dmap)
        root_id, preorder = self._validate_topology_and_sequences(tree_edges, sequences)
        ignored = {ch.upper() for ch in dmap.get("ignored_characters", ())}

        cumulative: dict[str, float] = {root_id: 0.0}
        pairwise_root: dict[str, float] = {root_id: 0.0}
        deficits: dict[str, float] = {root_id: 0.0}
        root_seq = sequences[root_id]

        children: dict[str, list[str]] = defaultdict(list)
        for node, parent in tree_edges.items():
            if parent is not None:
                children[parent].append(node)

        site_branch_mutations: dict[int, int] = defaultdict(int)
        site_shared_tip_counts: dict[int, int] = defaultdict(int)
        total_tree_events = 0

        def _tip_count(node_id: str) -> int:
            kids = children.get(node_id, [])
            if not kids:
                return 1
            return sum(_tip_count(k) for k in kids)

        for node_id in preorder:
            parent_id = tree_edges[node_id]
            if parent_id is None:
                continue
            parent_seq = sequences[parent_id]
            child_seq = sequences[node_id]
            branch_dist = _pair_distance(
                parent_seq, child_seq, dmap, consolidate_indels=True
            )
            cumulative[node_id] = cumulative[parent_id] + branch_dist
            pairwise_root[node_id] = _pair_distance(
                root_seq, child_seq, dmap, consolidate_indels=True
            )
            deficits[node_id] = cumulative[node_id] - pairwise_root[node_id]

            # Track discrete evolutionary events per site (collapsing contiguous indels)
            in_gap = False
            n_desc_tips = _tip_count(node_id)
            for site, (raw_a, raw_b) in enumerate(zip(parent_seq, child_seq)):
                base_a = raw_a.upper()
                base_b = raw_b.upper()
                if base_a == base_b or base_a in ignored or base_b in ignored:
                    in_gap = False
                    continue
                if base_a == "-" or base_b == "-":
                    if not in_gap:
                        in_gap = True
                        total_tree_events += 1
                        site_branch_mutations[site] += 1
                        site_shared_tip_counts[site] = max(
                            site_shared_tip_counts[site], n_desc_tips
                        )
                else:
                    in_gap = False
                    total_tree_events += 1
                    site_branch_mutations[site] += 1
                    site_shared_tip_counts[site] = max(
                        site_shared_tip_counts[site], n_desc_tips
                    )

        homoplasy_sites = sorted(
            site for site, n_branches in site_branch_mutations.items() if n_branches > 1
        )
        synapomorphy_sites = sorted(
            site
            for site, n_branches in site_branch_mutations.items()
            if n_branches == 1 and site_shared_tip_counts[site] >= 2
        )
        return {
            "cumulative_distances": cumulative,
            "pairwise_root_distances": pairwise_root,
            "saturation_deficits": deficits,
            "total_tree_events": total_tree_events,
            "homoplasy_sites": homoplasy_sites,
            "synapomorphy_sites": synapomorphy_sites,
        }


def _mutate(seq: str, edits: Sequence[tuple[int, str]]) -> str:
    chars = list(seq)
    for idx, base in edits:
        chars[idx] = base
    return "".join(chars)


def audit_evolutionary_distance_pipeline(
    auditor: Any | None = None,
) -> dict[str, Any]:
    """Run 6 evolutionary distance and molecular-clock invariants on benign controls."""
    engine = auditor if auditor is not None else DefensiveEvolutionaryDistanceAuditor()
    root_seq = SYNTHETIC_CLOCK_ROOT_SEQ
    invariants: list[dict[str, Any]] = []

    # 1. Reversion saturation recovery (Root -> Node1 -> TipRev: site 3 G->A->G, site 6 G->C->T)
    try:
        node1 = _mutate(root_seq, [(3, "A"), (6, "C")])
        tip_rev = _mutate(node1, [(3, "G"), (6, "T")])
        res1 = engine.analyze_tree(
            {"Root": None, "Node1": "Root", "TipRev": "Node1"},
            {"Root": root_seq, "Node1": node1, "TipRev": tip_rev},
        )
        cum_tip = float(res1["cumulative_distances"]["TipRev"])
        deficit = float(res1["saturation_deficits"]["TipRev"])
        inv1_ok = abs(cum_tip - 4.0) <= 1e-9 and abs(deficit - 3.0) <= 1e-9
        inv1_detail = f"cumulative={cum_tip}, saturation_deficit={deficit}"
    except Exception as exc:
        inv1_ok = False
        inv1_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "reversion_saturation_recovery",
            "passed": inv1_ok,
            "detail": inv1_detail,
        }
    )

    # 2. Homoplasy vs synapomorphy separation
    try:
        clade_a = _mutate(root_seq, [(9, "T")])
        clade_b = _mutate(root_seq, [(9, "T")])
        clade_c = _mutate(root_seq, [(12, "A")])
        res2 = engine.analyze_tree(
            {
                "Root": None,
                "CladeA": "Root",
                "TipA1": "CladeA",
                "TipA2": "CladeA",
                "CladeB": "Root",
                "TipB1": "CladeB",
                "CladeC": "Root",
                "TipC1": "CladeC",
                "TipC2": "CladeC",
            },
            {
                "Root": root_seq,
                "CladeA": clade_a,
                "TipA1": clade_a,
                "TipA2": clade_a,
                "CladeB": clade_b,
                "TipB1": clade_b,
                "CladeC": clade_c,
                "TipC1": clade_c,
                "TipC2": clade_c,
            },
        )
        inv2_ok = (
            res2.get("total_tree_events") == 3
            and res2.get("homoplasy_sites") == [9]
            and res2.get("synapomorphy_sites") == [12]
            and abs(float(res2["cumulative_distances"]["TipA1"]) - 1.0) <= 1e-9
        )
        inv2_detail = (
            f"events={res2.get('total_tree_events')}, "
            f"homoplasy={res2.get('homoplasy_sites')}, "
            f"synapomorphy={res2.get('synapomorphy_sites')}"
        )
    except Exception as exc:
        inv2_ok = False
        inv2_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "homoplasy_vs_synapomorphy_separation",
            "passed": inv2_ok,
            "detail": inv2_detail,
        }
    )

    # 3. Ambiguous call / dropout exclusion
    try:
        tip_amb = _mutate(root_seq, [(15, "N"), (16, "?")])
        res3 = engine.analyze_tree(
            {"Root": None, "TipAmb": "Root"},
            {"Root": root_seq, "TipAmb": tip_amb},
            distance_map={"default": 1.0, "ignored_characters": ["N", "?"], "map": {}},
        )
        dist3 = float(res3["cumulative_distances"]["TipAmb"])
        inv3_ok = abs(dist3 - 0.0) <= 1e-9
        inv3_detail = f"ambiguous_tip_distance={dist3}"
    except Exception as exc:
        inv3_ok = False
        inv3_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "ambiguous_call_exclusion",
            "passed": inv3_ok,
            "detail": inv3_detail,
        }
    )

    # 4. Contiguous multi-base indel consolidation
    try:
        tip_del = _mutate(root_seq, [(18, "-"), (19, "-"), (20, "-")])
        res4 = engine.analyze_tree(
            {"Root": None, "TipDel": "Root"},
            {"Root": root_seq, "TipDel": tip_del},
        )
        dist4 = float(res4["cumulative_distances"]["TipDel"])
        inv4_ok = abs(dist4 - 1.0) <= 1e-9
        inv4_detail = f"consolidated_3nt_indel_distance={dist4}"
    except Exception as exc:
        inv4_ok = False
        inv4_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "contiguous_indel_consolidation",
            "passed": inv4_ok,
            "detail": inv4_detail,
        }
    )

    # 5. Asymmetric directed antigenic map invariance & negative-weight guard
    try:
        esc_node = _mutate(root_seq, [(3, "A")])
        rev_tip = _mutate(esc_node, [(3, "G")])
        asym_map = {
            "default": 0.0,
            "map": {3: {("G", "A"): 1.25, ("A", "G"): 0.05}},
        }
        res5 = engine.analyze_tree(
            {"Root": None, "EscNode": "Root", "RevTip": "EscNode"},
            {"Root": root_seq, "EscNode": esc_node, "RevTip": rev_tip},
            distance_map=asym_map,
        )
        cum5 = float(res5["cumulative_distances"]["RevTip"])
        neg_rejected = False
        try:
            engine.analyze_tree(
                {"Root": None, "EscNode": "Root"},
                {"Root": root_seq, "EscNode": esc_node},
                distance_map={"default": 0.0, "map": {3: {("G", "A"): -0.5}}},
            )
        except ValueError:
            neg_rejected = True
        inv5_ok = abs(cum5 - 1.30) <= 1e-9 and neg_rejected
        inv5_detail = f"cumulative_asym={cum5:.4f}, negative_rejected={neg_rejected}"
    except Exception as exc:
        inv5_ok = False
        inv5_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "asymmetric_antigenic_map_invariance",
            "passed": inv5_ok,
            "detail": inv5_detail,
        }
    )

    # 6. Missing ancestral sequence & cyclic topology fail-closed check
    try:
        missing_caught = False
        cyclic_caught = False
        try:
            engine.analyze_tree(
                {"Root": None, "MissingNode": "Root"},
                {"Root": root_seq},
            )
        except EvolutionaryTopologyError:
            missing_caught = True
        try:
            engine.analyze_tree(
                {"Root": None, "NodeA": "NodeB", "NodeB": "NodeA"},
                {"Root": root_seq, "NodeA": root_seq, "NodeB": root_seq},
            )
        except EvolutionaryTopologyError:
            cyclic_caught = True
        inv6_ok = missing_caught and cyclic_caught
        inv6_detail = f"missing_caught={missing_caught}, cyclic_caught={cyclic_caught}"
    except Exception as exc:
        inv6_ok = False
        inv6_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "missing_ancestral_or_cyclic_topology_fail_closed",
            "passed": inv6_ok,
            "detail": inv6_detail,
        }
    )

    passed_count = sum(1 for inv in invariants if inv["passed"])
    return {
        "passed": passed_count == len(invariants),
        "passed_count": passed_count,
        "total_invariants": len(invariants),
        "invariants": invariants,
    }


def run_evolutionary_distance_self_check() -> dict[str, Any]:
    """Compare DefensiveEvolutionaryDistanceAuditor against NaivePairwiseRootDistanceCalculator."""
    hardened = audit_evolutionary_distance_pipeline(
        DefensiveEvolutionaryDistanceAuditor()
    )
    naive = audit_evolutionary_distance_pipeline(NaivePairwiseRootDistanceCalculator())
    caught = [inv["invariant"] for inv in naive["invariants"] if not inv["passed"]]
    return {
        "hardened_passed": hardened["passed"],
        "hardened": hardened,
        "naive_caught_invariants": caught,
        "naive_total_invariants": len(naive["invariants"]),
    }
