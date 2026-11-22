"""Pathogen clade surveillance and lineage-assignment invariant auditor.

Uses strictly benign synthetic housekeeping/reporter phylogeny templates (zero
pathogen sequences) to verify whether clade-assignment pipelines enforce:
1. Fail-closed handling when defined clades are missing from a tree (--error-on-missing-clade),
2. Acyclic single-parent clade inheritance validation,
3. Specificity-ordered derived subclade assignment independent of row order,
4. Strict 1-based to 0-based coordinate bounds (no 0 -> -1 wraparound),
5. Preorder branch-reversion mutation tracking, and
6. Mosaic/recombinant dual-lineage flagging instead of silent single-clade forcing.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

# Strictly benign synthetic housekeeping/reporter sequence (40 nt; zero pathogen sequences).
SYNTHETIC_CLADE_ROOT_SEQ = "ATGGCTAGCGATGCTAGCTAGTCGGATCGATCGATCGATC"

UNASSIGNED = "unassigned"


class CladeDefinitionError(ValueError):
    """Raised when clade inheritance or 1-based coordinate rules are violated."""


class MissingCladeError(RuntimeError):
    """Raised when --error-on-missing-clade is active and defined clades are absent."""


class DefensiveCladeAssigner:
    """Reference clade assigner satisfying all 6 pathogen surveillance invariants."""

    @staticmethod
    def validate_and_resolve_definitions(
        definitions: Mapping[str, Mapping[str, Any]],
        root_seq: str = SYNTHETIC_CLADE_ROOT_SEQ,
    ) -> dict[str, dict[str, Any]]:
        """Validate inheritance DAG, bounds, and return topological clade specs."""
        if not definitions:
            raise CladeDefinitionError("No clade definitions supplied.")

        root_norm = (root_seq or "").strip().upper()
        root_len = len(root_norm)
        if root_len == 0:
            raise CladeDefinitionError("Root sequence must be non-empty.")

        parents: dict[str, str | None] = {}
        direct_muts: dict[str, dict[tuple[str, int], str]] = {}

        for raw_name, spec in definitions.items():
            name = str(raw_name).strip()
            if not name:
                raise CladeDefinitionError("Clade name must be non-empty.")
            parent_raw = spec.get("parent")
            parent = str(parent_raw).strip() if parent_raw else None
            if parent == name:
                raise CladeDefinitionError(
                    f"Clade {name!r} cannot inherit from itself."
                )
            parents[name] = parent

            mut_map: dict[tuple[str, int], str] = {}
            for entry in spec.get("mutations", ()):
                gene, site, alt = entry
                if not isinstance(site, int) or isinstance(site, bool):
                    raise CladeDefinitionError(
                        f"Clade {name!r} site must be an integer, got {site!r}."
                    )
                if site < 1 or site > root_len:
                    raise CladeDefinitionError(
                        f"Clade {name!r} 1-based site {site} is out of bounds [1, {root_len}]."
                    )
                alt_norm = str(alt).strip().upper()
                if not alt_norm:
                    raise CladeDefinitionError(
                        f"Clade {name!r} has empty allele at site {site}."
                    )
                mut_map[(str(gene), site - 1)] = alt_norm
            direct_muts[name] = mut_map

        # Ensure every parent exists before cycle detection.
        for name, parent in parents.items():
            if parent is not None and parent not in parents:
                raise CladeDefinitionError(
                    f"Clade {name!r} inherits from undefined parent clade {parent!r}."
                )

        # Topological sort with cycle detection (0=unvisited, 1=visiting, 2=done).
        state: dict[str, int] = {name: 0 for name in parents}
        topo_order: list[str] = []
        depth: dict[str, int] = {}

        def _visit(node: str) -> int:
            if state[node] == 1:
                raise CladeDefinitionError(
                    f"Cyclic clade inheritance detected involving {node!r}."
                )
            if state[node] == 2:
                return depth[node]
            state[node] = 1
            par = parents[node]
            d = 0 if par is None else _visit(par) + 1
            state[node] = 2
            depth[node] = d
            topo_order.append(node)
            return d

        for clade_name in parents:
            _visit(clade_name)

        resolved: dict[str, dict[str, Any]] = {}
        for clade_name in topo_order:
            par = parents[clade_name]
            inherited = dict(resolved[par]["alleles"]) if par is not None else {}
            inherited.update(direct_muts[clade_name])
            resolved[clade_name] = {
                "parent": par,
                "depth": depth[clade_name],
                "alleles": inherited,
                "direct_alleles": direct_muts[clade_name],
            }
        return resolved

    @staticmethod
    def _parse_branch_mut(
        raw: str | tuple[str, int, str], root_len: int
    ) -> tuple[str, int, str]:
        if isinstance(raw, tuple):
            gene, site, alt = raw
            if site < 1 or site > root_len:
                raise CladeDefinitionError(
                    f"Mutation site {site} out of bounds [1, {root_len}]."
                )
            return str(gene), site - 1, str(alt).strip().upper()
        text = str(raw).strip().upper()
        if len(text) < 3 or not text[1:-1].isdigit():
            raise CladeDefinitionError(f"Malformed mutation token: {raw!r}.")
        site = int(text[1:-1])
        if site < 1 or site > root_len:
            raise CladeDefinitionError(
                f"Mutation site {site} out of bounds [1, {root_len}]."
            )
        return "nuc", site - 1, text[-1]

    def assign_tree_clades(
        self,
        definitions: Mapping[str, Mapping[str, Any]],
        nodes: Sequence[Mapping[str, Any]],
        root_seq: str = SYNTHETIC_CLADE_ROOT_SEQ,
        error_on_missing_clade: bool = False,
    ) -> dict[str, Any]:
        """Assign clades on a preorder tree while enforcing specificity and reversions."""
        resolved = self.validate_and_resolve_definitions(definitions, root_seq=root_seq)
        root_chars = list(root_seq.strip().upper())
        root_len = len(root_chars)

        node_seqs: dict[str, list[str]] = {}
        node_order: list[str] = []
        for node in nodes:
            nid = str(node["id"])
            parent_id = node.get("parent")
            base = (
                list(node_seqs[str(parent_id)])
                if parent_id is not None and str(parent_id) in node_seqs
                else list(root_chars)
            )
            for raw_mut in node.get("mutations", ()):
                gene, pos, alt = self._parse_branch_mut(raw_mut, root_len)
                if gene == "nuc":
                    base[pos] = alt
            node_seqs[nid] = base
            node_order.append(nid)

        # Evaluate most specific (deepest / largest allele set) clades first.
        ordered_clades = sorted(
            resolved.keys(),
            key=lambda cname: (
                resolved[cname]["depth"],
                len(resolved[cname]["alleles"]),
                cname,
            ),
            reverse=True,
        )

        membership: dict[str, str] = {}
        for nid in node_order:
            seq = node_seqs[nid]
            assigned = UNASSIGNED
            for cname in ordered_clades:
                alleles = resolved[cname]["alleles"]
                if all(
                    gene == "nuc" and seq[pos] == alt
                    for (gene, pos), alt in alleles.items()
                ):
                    assigned = cname
                    break
            membership[nid] = assigned

        found_clades = {val for val in membership.values() if val != UNASSIGNED}
        missing_clades = sorted(set(resolved.keys()) - found_clades)
        if error_on_missing_clade and missing_clades:
            raise MissingCladeError(
                "Clade(s) not found in tree: "
                + ", ".join(repr(c) for c in missing_clades)
            )

        return {
            "membership": membership,
            "missing_clades": missing_clades,
            "found_clades": sorted(found_clades),
        }

    def screen_recombinant(
        self,
        definitions: Mapping[str, Mapping[str, Any]],
        query_seq: str,
        root_seq: str = SYNTHETIC_CLADE_ROOT_SEQ,
    ) -> dict[str, Any]:
        """Flag chimeric sequences carrying disjoint blocks from unrelated lineages."""
        resolved = self.validate_and_resolve_definitions(definitions, root_seq=root_seq)
        qnorm = (query_seq or "").strip().upper()
        if len(qnorm) != len(root_seq.strip()):
            raise CladeDefinitionError(
                "Query length does not match root sequence length."
            )

        def _is_ancestor(anc: str, desc: str) -> bool:
            cur: str | None = desc
            while cur is not None:
                if cur == anc:
                    return True
                cur = resolved[cur]["parent"]
            return False

        # Check for partial or full multi-site support (>=2 hits) across unrelated lineages
        # BEFORE returning single-clade matches, so a mosaic carrying all of Lineage_X plus
        # a multi-site block from unrelated Lineage_Y is still flagged as a recombinant.
        support_counts: dict[str, int] = {}
        matched_positions: dict[str, set[int]] = {}
        for cname, spec in resolved.items():
            hits = {
                pos
                for (gene, pos), alt in spec["alleles"].items()
                if gene == "nuc" and qnorm[pos] == alt
            }
            if len(hits) >= 2:
                support_counts[cname] = len(hits)
                matched_positions[cname] = hits

        candidates = sorted(support_counts.keys())
        for i, c1 in enumerate(candidates):
            for c2 in candidates[i + 1 :]:
                if _is_ancestor(c1, c2) or _is_ancestor(c2, c1):
                    continue
                if matched_positions[c1].isdisjoint(matched_positions[c2]):
                    return {
                        "recombinant_candidate": True,
                        "assigned_clade": UNASSIGNED,
                        "recombinant_parents": tuple(sorted((c1, c2))),
                    }

        # Exact full-match clades (most specific first) when no mosaic dual-lineage signal exists.
        full_matches = [
            cname
            for cname in sorted(
                resolved,
                key=lambda c: (resolved[c]["depth"], len(resolved[c]["alleles"])),
                reverse=True,
            )
            if all(
                qnorm[pos] == alt
                for (gene, pos), alt in resolved[cname]["alleles"].items()
                if gene == "nuc"
            )
        ]
        if full_matches:
            return {
                "recombinant_candidate": False,
                "assigned_clade": full_matches[0],
                "recombinant_parents": (),
            }

        return {
            "recombinant_candidate": False,
            "assigned_clade": UNASSIGNED,
            "recombinant_parents": (),
        }

    def __call__(self, stage: str, payload: Mapping[str, Any]) -> Any:
        """Dispatch single-callable clade surveillance interface for auditing."""
        if stage == "validate_definitions":
            return self.validate_and_resolve_definitions(
                payload["definitions"],
                root_seq=payload.get("root_seq", SYNTHETIC_CLADE_ROOT_SEQ),
            )
        if stage == "assign_tree":
            return self.assign_tree_clades(
                payload["definitions"],
                payload["nodes"],
                root_seq=payload.get("root_seq", SYNTHETIC_CLADE_ROOT_SEQ),
                error_on_missing_clade=bool(
                    payload.get("error_on_missing_clade", False)
                ),
            )
        if stage == "screen_recombinant":
            return self.screen_recombinant(
                payload["definitions"],
                payload["query_seq"],
                root_seq=payload.get("root_seq", SYNTHETIC_CLADE_ROOT_SEQ),
            )
        raise ValueError(f"Unknown clade surveillance stage: {stage!r}")


def audit_clade_assignment_pipeline(
    pipeline_fn: Callable[[str, Mapping[str, Any]], Any],
) -> dict[str, Any]:
    """Audit a clade-assignment pipeline across 6 pathogen genomic invariants."""
    invariants: list[dict[str, Any]] = []
    root = SYNTHETIC_CLADE_ROOT_SEQ
    rchars = list(root)
    alt5 = "C" if rchars[4] != "C" else "A"
    alt10 = "G" if rchars[9] != "G" else "T"
    alt15 = "T" if rchars[14] != "T" else "C"

    # 1. Fail closed on missing/overwritten defined clade when flag is enabled
    try:
        defs_missing = {
            "Clade_24H": {"parent": None, "mutations": [("nuc", 5, alt5)]},
            "Clade_25C": {"parent": "Clade_24H", "mutations": [("nuc", 10, alt10)]},
        }
        nodes_missing = [
            {"id": "root", "parent": None, "mutations": []},
            {
                "id": "tip1",
                "parent": "root",
                "mutations": [f"{rchars[4]}5{alt5}", f"{rchars[9]}10{alt10}"],
            },
        ]
        try:
            pipeline_fn(
                "assign_tree",
                {
                    "definitions": defs_missing,
                    "nodes": nodes_missing,
                    "root_seq": root,
                    "error_on_missing_clade": True,
                },
            )
            inv1_ok = False
            inv1_detail = (
                "Silently succeeded when basal clade 24H had zero tips on tree."
            )
        except MissingCladeError:
            inv1_ok = True
            inv1_detail = "Raised MissingCladeError on overwritten Clade_24H."
    except Exception as exc:
        inv1_ok = False
        inv1_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "missing_defined_clade_fail_closed",
            "passed": inv1_ok,
            "detail": inv1_detail,
        }
    )

    # 2. Acyclic single-parent inheritance (reject self/cycle/undefined parent)
    try:
        rejected_cycle = False
        rejected_missing_parent = False
        try:
            pipeline_fn(
                "validate_definitions",
                {
                    "definitions": {
                        "A": {"parent": "B", "mutations": [("nuc", 5, alt5)]},
                        "B": {"parent": "A", "mutations": [("nuc", 10, alt10)]},
                    },
                    "root_seq": root,
                },
            )
        except CladeDefinitionError:
            rejected_cycle = True

        try:
            pipeline_fn(
                "validate_definitions",
                {
                    "definitions": {
                        "Child": {
                            "parent": "NonExistent",
                            "mutations": [("nuc", 5, alt5)],
                        }
                    },
                    "root_seq": root,
                },
            )
        except CladeDefinitionError:
            rejected_missing_parent = True

        inv2_ok = rejected_cycle and rejected_missing_parent
        inv2_detail = (
            f"cycle={rejected_cycle}, missing_parent={rejected_missing_parent}"
        )
    except Exception as exc:
        inv2_ok = False
        inv2_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "acyclic_single_parent_inheritance",
            "passed": inv2_ok,
            "detail": inv2_detail,
        }
    )

    # 3. Derived subclade specificity over basal clade independent of input ordering
    try:
        # Deliberately place derived Clade_1A BEFORE basal Clade_1
        defs_order = {
            "Clade_1A": {"parent": "Clade_1", "mutations": [("nuc", 10, alt10)]},
            "Clade_1": {"parent": None, "mutations": [("nuc", 5, alt5)]},
        }
        nodes_order = [
            {"id": "root", "parent": None, "mutations": []},
            {"id": "basal_tip", "parent": "root", "mutations": [f"{rchars[4]}5{alt5}"]},
            {
                "id": "derived_tip",
                "parent": "basal_tip",
                "mutations": [f"{rchars[9]}10{alt10}"],
            },
        ]
        res_order = pipeline_fn(
            "assign_tree",
            {
                "definitions": defs_order,
                "nodes": nodes_order,
                "root_seq": root,
                "error_on_missing_clade": True,
            },
        )
        mem = res_order["membership"]
        inv3_ok = (
            mem.get("basal_tip") == "Clade_1" and mem.get("derived_tip") == "Clade_1A"
        )
        inv3_detail = f"membership={mem!r}"
    except Exception as exc:
        inv3_ok = False
        inv3_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "derived_subclade_specificity_over_basal_clade",
            "passed": inv3_ok,
            "detail": inv3_detail,
        }
    )

    # 4. 1-based to 0-based coordinate bounds (no 0 -> -1 wraparound)
    try:
        try:
            pipeline_fn(
                "validate_definitions",
                {
                    "definitions": {
                        "BadZero": {"parent": None, "mutations": [("nuc", 0, "C")]},
                    },
                    "root_seq": root,
                },
            )
            inv4_ok = False
            inv4_detail = (
                "Accepted 1-based site=0 without raising CladeDefinitionError."
            )
        except CladeDefinitionError:
            inv4_ok = True
            inv4_detail = "Rejected site=0 before 0-based indexing wraparound."
    except Exception as exc:
        inv4_ok = False
        inv4_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "one_based_to_zero_based_coordinate_parity",
            "passed": inv4_ok,
            "detail": inv4_detail,
        }
    )

    # 5. Reversion-aware preorder mutation tracking (G10A after A10G restores root allele)
    try:
        defs_rev = {
            "Clade_Mut10": {"parent": None, "mutations": [("nuc", 10, alt10)]},
        }
        nodes_rev = [
            {"id": "root", "parent": None, "mutations": []},
            {
                "id": "internal",
                "parent": "root",
                "mutations": [f"{rchars[9]}10{alt10}"],
            },
            {
                "id": "reverted_tip",
                "parent": "internal",
                "mutations": [f"{alt10}10{rchars[9]}"],
            },
        ]
        res_rev = pipeline_fn(
            "assign_tree",
            {
                "definitions": defs_rev,
                "nodes": nodes_rev,
                "root_seq": root,
                "error_on_missing_clade": False,
            },
        )
        mem_rev = res_rev["membership"]
        inv5_ok = (
            mem_rev.get("internal") == "Clade_Mut10"
            and mem_rev.get("reverted_tip") == UNASSIGNED
        )
        inv5_detail = f"membership={mem_rev!r}"
    except Exception as exc:
        inv5_ok = False
        inv5_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "reversion_aware_mutation_state_tracking",
            "passed": inv5_ok,
            "detail": inv5_detail,
        }
    )

    # 6. Mosaic / recombinant dual-lineage flagging on disjoint mutation blocks
    try:
        alt28 = "A" if rchars[27] != "A" else "G"
        alt32 = "C" if rchars[31] != "C" else "T"
        defs_rec = {
            "Lineage_X": {
                "parent": None,
                "mutations": [("nuc", 5, alt5), ("nuc", 15, alt15)],
            },
            "Lineage_Y": {
                "parent": None,
                "mutations": [("nuc", 28, alt28), ("nuc", 32, alt32)],
            },
        }
        mosaic = list(root)
        mosaic[4] = alt5
        mosaic[14] = alt15
        mosaic[27] = alt28
        mosaic[31] = alt32
        res_rec = pipeline_fn(
            "screen_recombinant",
            {"definitions": defs_rec, "query_seq": "".join(mosaic), "root_seq": root},
        )
        inv6_ok = (
            isinstance(res_rec, Mapping)
            and res_rec.get("recombinant_candidate") is True
            and tuple(res_rec.get("recombinant_parents", ()))
            == ("Lineage_X", "Lineage_Y")
        )
        inv6_detail = f"res={res_rec!r}"
    except Exception as exc:
        inv6_ok = False
        inv6_detail = f"Unexpected error: {exc}"
    invariants.append(
        {
            "invariant": "recombinant_dual_lineage_flagging",
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


def _naive_clade_assigner(stage: str, payload: Mapping[str, Any]) -> Any:
    """Reference naive clade assigner exhibiting all 6 surveillance defects."""
    if stage == "validate_definitions":
        # Defect 2 & 4: skips DAG cycle check, missing parent check, and wraps site=0 -> -1.
        out = {}
        for cname, spec in payload["definitions"].items():
            out[cname] = {
                "alleles": {(g, int(s) - 1): a for g, s, a in spec.get("mutations", ())}
            }
        return out

    if stage == "assign_tree":
        # Defect 1, 3, 5: ignores error_on_missing_clade, overwrites by insertion order,
        # and checks if mutation ever occurred on branch path without tracking reversions.
        root = payload["root_seq"]
        rchars = list(root)
        defs = payload["definitions"]
        node_seen: dict[str, set[tuple[int, str]]] = {}
        membership: dict[str, str] = {}
        for node in payload["nodes"]:
            nid = str(node["id"])
            par = node.get("parent")
            seen = set(node_seen.get(str(par), set())) if par is not None else set()
            for raw_mut in node.get("mutations", ()):
                pos = int(str(raw_mut)[1:-1]) - 1
                alt = str(raw_mut)[-1].upper()
                seen.add((pos, alt))
            node_seen[nid] = seen

            assigned = UNASSIGNED
            # Defect 3: last matching definition wins in dict order (basal overwrites derived)
            for cname, spec in defs.items():
                muts = spec.get("mutations", ())
                if all((s - 1, a) in seen or (rchars[s - 1] == a) for _g, s, a in muts):
                    assigned = cname
            membership[nid] = assigned
        return {"membership": membership, "missing_clades": [], "found_clades": []}

    if stage == "screen_recombinant":
        # Defect 6: silently assigns first matching clade without checking chimeric blocks.
        first = next(iter(payload["definitions"].keys()), UNASSIGNED)
        return {
            "recombinant_candidate": False,
            "assigned_clade": first,
            "recombinant_parents": (),
        }

    raise ValueError(f"Unknown stage: {stage!r}")


def run_clade_assignment_self_check() -> dict[str, Any]:
    """Compare hardened DefensiveCladeAssigner against a naive clade assigner."""
    hardened = audit_clade_assignment_pipeline(DefensiveCladeAssigner())
    naive = audit_clade_assignment_pipeline(_naive_clade_assigner)
    caught = [inv["invariant"] for inv in naive["invariants"] if not inv["passed"]]
    return {
        "hardened_passed": hardened["passed"],
        "hardened": hardened,
        "naive_caught_invariants": caught,
        "naive_total_invariants": naive["total_invariants"],
    }
