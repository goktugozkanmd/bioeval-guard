# bioeval-guard

Deterministic dataset hygiene, biological sequence verification, and zero-question shortcut auditing for frontier AI biosecurity and biology benchmarks (`LAB-Bench`, `WMDP`, `BixBench`, and `Inspect AI` pipelines).

## Why This Exists

Frontier biosecurity evaluations inform model release gates, safeguards thresholds, and safety claims. Auditing **3,428 public evaluation items** across `futurehouse/lab-bench` (`5c77cec6`), `cais/wmdp` (`7125571f`), and `futurehouse/BixBench` (`f8cc3bdc`) with **zero LLM calls (`$0` API)** and strictly out-of-fold / Holm-Bonferroni controls revealed both severe shortcut vulnerabilities and surprisingly clean control subsets:

| Benchmark / Subset | N | Chance | Centroid | Longest | 5-Fold OOF | Holm `p_adj` | Key Integrity Findings |
|---|---:|---:|---:|---:|---:|---:|---|
| `WMDP/bio` | 1,273 | 25.0% | 21.3% | **46.1%** | **45.8%** | **`1.16e-58`** (`Longest`) / `5.72e-57` (`OOF`) | Longest-option length bias (`+21.1 pp`), `14` ideal-only meta-options (`All of the above` `11/12` ideal) |
| `LAB-Bench/SeqQA` | 600 | 24.7% | **44.5%** | 27.3% | **41.2%** | **`3.61e-25`** (`Centroid`) | `PCR_ORF` subset (`N=480`): **49.6%** Centroid; `non_PCR_ORF` (`N=120`): 20.8% (`clean`). `40/40` sense reverse primers (`PCR-seq-enzprimers`), `8` isoschizomer collisions, `1` exact `ideal==distractor` (`ded3f6ac`) |
| `LAB-Bench/DbQA` | 520 | 25.0% | **35.4%** | 23.5% | **37.7%** | **`1.23e-09`** (`OOF`) | `variant_multi_sequence_task` (`N=80`): **100.0% (`80/80`)** edit-distance centroid (single-AA star-graph perturbations) |
| `LAB-Bench/LitQA2` | 199 | 24.4% | 26.6% | 24.6% | 26.6% | `1.00` (clean shortcuts) | **`5/199` literal `"null"` `ideal` answers**, `2` comma-split distractor shards (`'12,300'` vs `'4','900'...`), `2` duplicate distractors |
| `WMDP/chem` | 408 | 25.0% | 29.7% | 30.9% | 30.1% | `0.114` (n.s.) | Passes Holm-Bonferroni threshold after correction; `2` ideal-only meta-options |
| `BixBench` | 205 | 25.0% | 25.9% | 24.4% | 25.4% | `1.00` (clean shortcuts) | Strongest option-geometry resistance; `2` `str_verifier` notation mismatches, `6` overlapping `range_verifier` boundaries |

## Installation

```bash
pip install -e .
```

## Quickstart

```bash
bioeval-guard summary datasets/wmdp_bio.parquet datasets/lab_bench_SeqQA.parquet datasets/lab_bench_DbQA.parquet datasets/BixBench.jsonl
```

### Inspect AI Pre-Filter & Register-Ready Usage

```python
from bioeval_guard.inspect_adapter import sanitize_mcq_dataset, question_blind_centroid_baseline
from bioeval_guard.inspect_tasks import filter_clean_lab_bench_records, nan_safe_precision_metric

clean_records, hygiene_report = sanitize_mcq_dataset(records, dataset_name="SeqQA")
```

### v0.2.0 Grader, Parser & Defensive DNA-Screening Symmetry Auditors

```bash
bioeval-guard self-check
```

`bioeval_guard.grader_audit` adds deterministic zero-API probes for:
- **Safety judge callables** (`audit_safety_judge_callable`): detects case-sensitive `"Unsafe"`/`"UNSAFE"` fallthrough-to-safe bugs, code-fenced JSON, and empty refusal completions.
- **XML `<answer>` parsers** (`audit_answer_tag_parser`): detects `str.find()` negative-slice corruption on untagged/unclosed prose and CoT prompt-echo collisions.
- **Numeric string graders** (`audit_numeric_string_verifier`): flags sign/decimal/exponent stripping (`-4.96` == `4.96`) and empty-string substring false positives.
- **Selective precision metrics** (`audit_selective_precision_metric`): distinguishes 100%-abstention (`NaN` precision, `0.0` coverage) from 100%-incorrect (`0.0` precision, `1.0` coverage).
- **Defensive DNA screener invariance** (`audit_dna_screener_symmetry`): verifies reverse-complement, FASTA/CRLF, and window-shift parity using strictly benign synthetic oligos (zero pathogen sequences).

### v0.4.0 Defensive Sequence-Screening Invariants & MCQ Option-Text Hijack Auditor

```bash
bioeval-guard screen-audit
bioeval-guard cluster-audit capsules.jsonl
```

- **6-Invariant Sequence-Screening Auditor** (`bioeval_guard.sequence_screening_audit.audit_sequence_screener` & `DefensiveKmerScreener`): tests strand symmetry, RNA uracil/case normalization, circular origin wrap-around, tiled fragment boundary assembly, IUPAC ambiguity resilience, and malformed FASTA header safety on strictly benign synthetic linker oligos.
- **MCQ Option-Text Inversion Auditor** (`audit_mcq_option_text_grader` & `hardened_mcq_option_grader`): catches `location="end"` option-text hijack vulnerabilities (e.g., `"C. extrachromosomal dna"` matching target `"A"`).
- **Cluster-Robust Benchmark SE/ICC/DEFF Auditor** (`audit_clustered_benchmark_se`): quantifies cluster-inflated precision on correlated multi-question capsules (`BixBench`, `LAB-Bench`).

### v0.5.0 Pathogen Metagenomic Surveillance & Unparseable Judge NaN Auditors

```bash
bioeval-guard surveillance-audit
```

- **6-Invariant Metagenomic Surveillance Auditor** (`bioeval_guard.metagenomic_surveillance_audit` & `DefensiveMetagenomicScreener`): audits host-chimera junction retention (`>=31 bp` non-host overhangs), strict zero-padded multiplex barcode canonicalization, fail-closed sample-sheet file existence checks, LCA alert-candidate preservation, spike-in compositional invariance under `100x` flora bloom, and paired-mate strand symmetry on strictly benign synthetic controls.
- **Unparseable Judge NaN Auditor** (`audit_unparseable_judge_nan_handling` & `unparseable_safe_nan_judge_score`): catches empty/content-filtered/malformed judge completions silently collapsing to `0.0` instead of `NaN`.

### v0.6.0 Phylodynamic Surveillance & Truncated-Completion Gate Auditors

```bash
bioeval-guard phylo-audit
```

- **6-Invariant Phylodynamic Surveillance Auditor** (`bioeval_guard.phylodynamic_surveillance_audit` & `DefensivePhylodynamicScreener`): audits sequence/metadata index parity, IUPAC/dropout mutation masking, incomplete collection-date uncertainty bounds, frame-preserving codon translation across deletions and 4-fold ambiguous codons, zero-length polytomy tie guarding, and Horvitz-Thompson sampling-depletion invariance on strictly benign synthetic housekeeping templates.
- **Truncated / Empty Completion Gate Auditor** (`audit_truncated_empty_completion_handling` & `truncation_safe_answer_gate`): strips `<think>`/`<reasoning>` traces and rejects `max_tokens`/`length`/`content_filter` empty outputs instead of passing `""` to downstream safety judges.

## Author & Contact

- **Author:** Göktuğ Özkan, MD (`drozkan2@gmail.com`, ORCID: `0000-0002-5022-9124`)
- **License:** MIT
