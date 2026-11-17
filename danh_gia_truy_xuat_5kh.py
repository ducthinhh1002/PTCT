"""Reproduce a file-level retrieval evaluation using frozen RepoReasoner traces.

Run with Python 3.11+: python -m pip install tiktoken==0.14.0
     python danh_gia_truy_xuat_5kh.py
Source archives are read as text. No repository code or tests are executed.
"""

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import statistics
import sys
import tempfile
import tokenize
from urllib.request import urlopen
import zipfile

import tiktoken


BENCHMARK = "bc69c086378d163246e7c82d85645c1699c8ba12"
SOURCES = {
    "climlab": ("climlab/climlab", "da1418b2579d25547032408cb3d4a84a2dcb874b"),
    "fluids": ("CalebBell/fluids", "61fa64d41c51721584d4eb751f087e6a8ddf3720"),
    "geopandas": ("geopandas/geopandas", "d92ff77398e01f099534f187de62c264db41f266"),
    "joblib": ("joblib/joblib", "0672e76ad10e9f463221a58fb7bffa127287a4ba"),
    "phonopy": ("phonopy/phonopy", "b13689331d61479accd7cbbc348b5daeeefcb2e2"),
    "pulp": ("coin-or/pulp", "ba4a17f0a3f29b7ea5bd02d17c1ff1f4c9466823"),
    "pykalman": ("pykalman/pykalman", "dedb0b43a392ac1bda7813f79766fd7ef8819ed9"),
    "seaborn": ("mwaskom/seaborn", "7001ebe72423238e99c0434a2ef0a0ebc9cb55c1"),
    "sfe": ("sfepy/sfepy", "83cdfd9302684df7a7e7bf88a2543c3731c5d8d4"),
    "stellargraph": ("stellargraph/stellargraph", "3c2c8c18ab4c5c16660f350d8e23d7dc39e738de"),
}
BUDGETS = (4096, 8192, 16384)
METHODS = ("bm25", "imports_1", "imports_3", "calls_1", "calls_3")
TOKENIZER = "o200k_base"
WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
WORDS = re.compile(r"[a-z]+|[0-9]+")


def fetch(url, target):
    if not target.exists():
        with urlopen(url, timeout=90) as response:
            content = response.read()
        staging = target.with_suffix(target.suffix + ".tmp")
        staging.write_bytes(content)
        staging.replace(target)
    return target


def records(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sources(name, cache):
    repository, revision = SOURCES[name]
    folder = cache / name
    folder.mkdir(parents=True, exist_ok=True)
    base = f"https://raw.githubusercontent.com/DeepSoftwareAnalytics/RepoReasoner/{BENCHMARK}"
    tasks_url = f"{base}/Data_RepoReasoner/original/{name}.jsonl"
    traces_url = f"{base}/output_results/{name}/report_functions.jsonl"
    tasks = records(fetch(tasks_url, folder / "cases.jsonl"))
    traces = records(fetch(traces_url, folder / "traces.jsonl"))
    archive = fetch(f"https://codeload.github.com/{repository}/zip/{revision}", folder / "source.zip")
    texts = {}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.endswith(".py"):
                prefix, path = member.split("/", 1)
                if not prefix.endswith(f"-{revision}"):
                    raise ValueError(f"Unexpected GitHub archive revision for {name}.")
                raw = bundle.read(member)
                try:
                    encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
                    texts[path] = raw.decode(encoding)
                except (SyntaxError, UnicodeDecodeError):
                    continue
    return tasks, traces, texts


def words(text):
    return WORDS.findall(WORD_BOUNDARY.sub(" ", text).lower())


def parse(text):
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def module_name(path):
    parts = PurePosixPath(path).with_suffix("").parts
    if parts[0] == "src":
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def attribute_parts(node):
    suffix = []
    while isinstance(node, ast.Attribute):
        suffix.append(node.attr)
        node = node.value
    return [node.id, *reversed(suffix)] if isinstance(node, ast.Name) else []


class FileIndex:
    def __init__(self, texts, encoder):
        self.trees = {path: parse(text) for path, text in texts.items()}
        self.cost = {path: len(encoder.encode_ordinary(f"# {path}\n{text}\n\n"))
                     for path, text in texts.items()}
        self.terms = {path: Counter(words(f"{path}\n{text}")) for path, text in texts.items()}
        self.length = {path: sum(counts.values()) for path, counts in self.terms.items()}
        self.df = Counter(term for counts in self.terms.values() for term in counts)
        self.total_length = sum(self.length.values())
        aliases = defaultdict(list)
        for path in sorted(texts):
            aliases[module_name(path)].append(path)
        self.modules = {module: paths[0] for module, paths in aliases.items() if module and len(paths) == 1}
        self.exports = {module: self.bindings(self.trees[path].body, path)
                        for module, path in self.modules.items() if self.trees[path] is not None}
        self.import_edges, self.call_edges = {}, {}
        for path, tree in self.trees.items():
            imports, calls = self.edges(tree, path)
            self.import_edges[path] = imports - {path}
            self.call_edges[path] = calls - {path}

    def bindings(self, nodes, path):
        module = module_name(path)
        package = module if path.endswith("/__init__.py") else module.rpartition(".")[0]
        bindings = {}
        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    first = alias.name.split(".", 1)[0]
                    bindings[alias.asname or first] = alias.name if alias.asname else first
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    scope = package.split(".") if package else []
                    climb = node.level - 1
                    if climb >= len(scope):
                        continue
                    base = ".".join([*scope[:len(scope) - climb], *([base] if base else [])])
                for alias in node.names:
                    if base and alias.name != "*":
                        bindings[alias.asname or alias.name] = f"{base}.{alias.name}"
        return bindings

    def resolve(self, name, seen=frozenset()):
        if name in seen:
            return None
        parts = name.split(".")
        for count in range(len(parts), 0, -1):
            module = ".".join(parts[:count])
            if module not in self.modules:
                continue
            if count < len(parts):
                exported = self.exports.get(module, {}).get(parts[count])
                if exported:
                    target = ".".join([exported, *parts[count + 1:]])
                    resolved = self.resolve(target, seen | {name})
                    if resolved:
                        return resolved
            return self.modules[module]
        return None

    def edges(self, tree, path):
        if tree is None:
            return set(), set()
        nodes = list(ast.walk(tree))
        bound = self.bindings(nodes, path)
        destinations = set(bound.values())
        destinations.update(alias.name for node in nodes if isinstance(node, ast.Import) for alias in node.names)
        imported = {target for name in destinations if (target := self.resolve(name))}
        called = set()
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            parts = attribute_parts(node.func)
            if parts and parts[0] in bound:
                name = ".".join([bound[parts[0]], *parts[1:]])
                target = self.resolve(name)
                if target:
                    called.add(target)
        return imported, called

    def bm25(self, query, source):
        candidates = [path for path in self.terms if path != source]
        size = len(candidates)
        avg_length = (self.total_length - self.length[source]) / size
        source_terms = self.terms[source]
        unique = set(words(query))
        idf = {}
        for term in unique:
            df = self.df[term] - (term in source_terms)
            idf[term] = math.log1p((size - df + 0.5) / (df + 0.5))
        scores = {}
        for path in candidates:
            counts = self.terms[path]
            norm = 1.5 * (0.25 + 0.75 * self.length[path] / avg_length)
            scores[path] = math.fsum(idf[term] * counts[term] * 2.5 / (counts[term] + norm)
                                     for term in sorted(unique & counts.keys()))
        return sorted(candidates, key=lambda path: (-round(scores[path], 12), path))


def expand(ranked, starts, edges, depth, source):
    visited = {source}
    distance = {}
    frontier = starts - visited
    for step in range(1, depth + 1):
        distance.update(dict.fromkeys(frontier, step))
        visited.update(frontier)
        frontier = {child for node in frontier for child in edges[node]} - visited
    return sorted(ranked, key=lambda path: distance.get(path, math.inf))


def choose(ranked, cost, budget):
    selected = []
    used = 0
    for path in ranked:
        if used + cost[path] <= budget:
            selected.append(path)
            used += cost[path]
    return selected, used


def function_key(row):
    name = row["test_function"].split("[", 1)[0].replace("::", ".")
    return row["test_file"], name


def test_ast_matches(task, tree):
    original = parse(task["code"])
    if tree is None or original is None:
        return False
    expected = next((node for node in original.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    klass = task.get("classname")
    candidates = tree.body
    if klass:
        owner = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == klass), None)
        if owner is None:
            return False
        candidates = owner.body
    source = next((node for node in candidates if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   and node.name == task["funcname"]), None)
    return expected is not None and source is not None and ast.dump(expected) == ast.dump(source)


def evaluate(name, cache):
    tasks, traces, texts = load_sources(name, cache)
    index = FileIndex(texts, tiktoken.get_encoding(TOKENIZER))
    labels = defaultdict(set)
    for trace in traces:
        labels[function_key(trace)].update(item["file"] for item in trace["dependencies"])
    audit, results = Counter(), []
    evaluated = set()
    for task in tasks:
        audit["published"] += 1
        source = task["testpath"]
        function = ".".join(filter(None, (task.get("classname"), task["funcname"])))
        key = (source, function)
        if key[0] not in texts:
            audit["missing_source"] += 1
            continue
        if key in evaluated:
            audit["repeated_test"] += 1
            continue
        if not test_ast_matches(task, index.trees[source]):
            audit["test_ast_mismatch"] += 1
            continue
        truth = labels[key] - {source}
        if not truth:
            audit["missing_trace"] += 1
            continue
        if not truth <= texts.keys():
            audit["missing_dependency"] += 1
            continue
        evaluated.add(key)
        audit["included"] += 1
        query = "\n".join([*task["imports"], task["masked_code"]])
        start_imports, start_calls = index.edges(parse(query), source)
        baseline = index.bm25(query, source)
        rankings = {"bm25": baseline}
        for label, start, edges in (("imports", start_imports, index.import_edges),
                                    ("calls", start_calls, index.call_edges)):
            for depth in (1, 3):
                rankings[f"{label}_{depth}"] = expand(baseline, start, edges, depth, source)
        gold_tokens = sum(index.cost[path] for path in truth)
        cheapest_gold_first = sorted(truth, key=lambda path: (index.cost[path], path))
        for budget in BUDGETS:
            oracle_picked, _ = choose(cheapest_gold_first, index.cost, budget)
            ceiling = len(oracle_picked) / len(truth)
            for method, ranked in rankings.items():
                picked, used = choose(ranked, index.cost, budget)
                assert used <= budget and source not in picked and len(picked) == len(set(picked))
                hits = len(truth.intersection(picked))
                assert hits <= len(oracle_picked)
                results.append({
                    "repo": name, "task": task["task_id"], "test": function, "source": source,
                    "method": method, "budget": budget, "used_tokens": used,
                    "gold_tokens": gold_tokens, "gold_fit": int(gold_tokens <= budget),
                    "oracle_recall_ceiling": ceiling,
                    "gold_files": len(truth), "retrieved_files": len(picked), "hits": hits,
                    "recall": hits / len(truth), "precision": hits / len(picked) if picked else 0.0,
                    "complete": int(hits == len(truth)), "gold_paths": json.dumps(sorted(truth)),
                    "retrieved_paths": json.dumps(picked),
                })
    return {"repo": name, "candidate_files": len(texts), **audit}, results


def aggregate(rows):
    group = defaultdict(list)
    for row in rows:
        group[row["budget"], row["method"]].append(row)
    for (budget, method), items in sorted(group.items()):
        repositories = defaultdict(list)
        for row in items:
            repositories[row["repo"]].append(row)
        grouped_means = [statistics.mean(row["recall"] for row in repo) for repo in repositories.values()]
        fit = [row for row in items if row["gold_fit"]]
        yield {"budget": budget, "method": method, "n": len(items), "repos": len(repositories),
               "recall": statistics.mean(row["recall"] for row in items),
               "repo_macro_recall": statistics.mean(grouped_means),
               "precision": statistics.mean(row["precision"] for row in items),
               "complete": statistics.mean(row["complete"] for row in items),
               "oracle_fit": len(fit) / len(items),
               "oracle_recall_ceiling": statistics.mean(row["oracle_recall_ceiling"] for row in items),
               "complete_when_fit": statistics.mean(row["complete"] for row in fit) if fit else None,
               "mean_tokens": statistics.mean(row["used_tokens"] for row in items)}


def paired(rows):
    keys = {(r["budget"], r["method"], r["repo"], r["task"]): r for r in rows}
    comparisons = (("calls_3", "bm25"), ("calls_3", "calls_1"),
                   ("calls_3", "imports_1"), ("calls_3", "imports_3"), ("imports_3", "bm25"))
    for budget in BUDGETS:
        for method, baseline in comparisons:
            repo_deltas = defaultdict(list)
            for row in rows:
                if row["budget"] == budget and row["method"] == method:
                    control = keys[budget, baseline, row["repo"], row["task"]]
                    repo_deltas[row["repo"]].append(row["recall"] - control["recall"])
            deltas = [d for values in repo_deltas.values() for d in values]
            means = [statistics.mean(values) for values in repo_deltas.values()]
            rng = random.Random(5440)
            samples = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(5000))
            yield {"budget": budget, "method": method, "baseline": baseline, "n": len(deltas),
                   "case_mean_delta": statistics.mean(deltas), "repo_macro_delta": statistics.mean(means),
                   "repo_ci_low": samples[124], "repo_ci_high": samples[4874],
                   "cases_better": sum(delta > 0 for delta in deltas),
                   "cases_worse": sum(delta < 0 for delta in deltas),
                   "repos_better": sum(delta > 0 for delta in means),
                   "repos_worse": sum(delta < 0 for delta in means)}


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "ptct_reporeasoner")
    cli.add_argument("--output", type=Path, default=Path("ket_qua_truy_xuat_5KH"))
    options = cli.parse_args()
    audits, all_rows = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for audit, rows in pool.map(lambda repo: evaluate(repo, options.cache), SOURCES):
            audits.append(audit)
            all_rows.extend(rows)
            print(f"{audit['repo']}: {audit.get('included', 0)}/{audit['published']} tests", flush=True)
    if not all_rows:
        raise RuntimeError("No matching traces and source snapshots.")
    options.output.mkdir(parents=True, exist_ok=True)
    write_csv(options.output / "screening.csv", audits)
    write_csv(options.output / "per_case.csv", all_rows)
    write_csv(options.output / "summary.csv", aggregate(all_rows))
    write_csv(options.output / "paired.csv", paired(all_rows))
    write_csv(options.output / "per_repo.csv", (
        {"repo": name, **item} for name in SOURCES
        for item in aggregate(row for row in all_rows if row["repo"] == name)
    ))
    manifest = {
        "benchmark_commit": BENCHMARK, "budgets": BUDGETS,
        "python": sys.version.split()[0], "tokenizer": TOKENIZER,
        "tiktoken_version": version("tiktoken"),
        "sources": {
            name: {"repository": repository, "revision": revision, "sha256": {
                file: file_digest(options.cache / name / file)
                for file in ("cases.jsonl", "traces.jsonl", "source.zip")
            }} for name, (repository, revision) in SOURCES.items()
        },
    }
    (options.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
