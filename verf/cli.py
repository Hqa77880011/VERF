import argparse
import copy
import heapq
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

from .method import VersionedFuzzy


def generate(args):
    rng = np.random.default_rng(args.seed)
    x = rng.normal(size=(args.samples, args.features)).astype(np.float32)
    tool = rng.integers(args.tools, size=args.samples)
    phase = np.minimum(np.arange(args.samples) // args.drift_interval, 3)
    directions = rng.normal(size=(3, args.features))
    directions = np.concatenate([directions, directions[:1]], 0)
    score = np.einsum("ij,ij->i", x, directions[phase]) + 0.2 * tool
    y = (score > args.rare_threshold).astype(int)
    delay = np.full(args.samples, args.delay, dtype=int)
    if args.delay_kind == "lognormal":
        delay = np.maximum(0, np.rint(rng.lognormal(np.log(max(args.delay, 1)), 0.5, args.samples))).astype(int)
    elif args.delay_kind == "fault_dependent":
        delay = np.where(y == 1, args.delay // 4, args.delay).astype(int)
    audit_x = rng.normal(size=(4, 1000, args.features)).astype(np.float32)
    audit_tool = rng.integers(args.tools, size=(4, 1000))
    audit_y = (np.einsum("pnd,pd->pn", audit_x, directions) + 0.2 * audit_tool > args.rare_threshold).astype(int)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, x=x, y=y, tool=tool, time=np.arange(args.samples), delay=delay,
                        drift=np.flatnonzero(np.diff(phase, prepend=phase[0])), phase=phase,
                        audit_x=audit_x, audit_y=audit_y, audit_tool=audit_tool)


def prepare(args):
    frame = pd.read_csv(args.csv).sort_values("time", kind="stable")
    if (frame["delay"] < 0).any() or frame["time"].duplicated().any():
        raise ValueError("Nonnegative delays and unique chronological indices required")
    x = frame[args.features].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=int)
    if sorted(np.unique(y)) != list(range(int(y.max()) + 1)):
        raise ValueError("Labels must be consecutive integers starting at zero")
    tool = pd.factorize(frame["tool"], sort=True)[0]
    time_index = frame["time"].to_numpy(dtype=int)
    delay = frame["delay"].to_numpy(dtype=int)
    if not np.isfinite(x).all():
        raise ValueError("Impute or mask sensor features using warmup-only statistics before export")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, y=y, tool=tool, time=time_index, delay=delay,
                        drift=np.flatnonzero(frame["drift"].to_numpy() == 1) if "drift" in frame else np.empty(0, dtype=int))


def recovery(predicted, target, drifts, warmup, batch_size, tolerance):
    answer = []
    correct = np.asarray(predicted) == target
    for drift in drifts:
        local = int(drift) - warmup
        if local <= 0 or local >= len(correct):
            continue
        reference = correct[max(0, local - 5 * batch_size):local].mean()
        recovered = None
        for start in range(local, len(correct) - 3 * batch_size + 1, batch_size):
            if all(correct[start + j * batch_size:start + (j + 1) * batch_size].mean() >= reference - tolerance for j in range(3)):
                recovered = (start - local) // batch_size
                break
        answer.append(dict(drift_index=int(drift), reference_accuracy=float(reference), recovery_batches=recovered))
    return answer


def run(args):
    with np.load(args.data, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    x, y, tool = data["x"], data["y"], data["tool"]
    if not 0 < args.warmup < len(x) or len(np.unique(y[:args.warmup])) != len(np.unique(y)):
        raise ValueError("Warmup must be shorter than stream and contain all classes")
    mean, scale = x[:args.warmup].mean(0), x[:args.warmup].std(0).clip(1e-6)
    x = (x - mean) / scale
    model = VersionedFuzzy(x[:args.warmup], y[:args.warmup], tool[:args.warmup], args)
    oracle_args = copy.copy(args)
    oracle_args.variant = "no_versions"
    oracle_args.minimum_feedback = args.minimum_feedback
    oracle = VersionedFuzzy(x[:args.warmup], y[:args.warmup], tool[:args.warmup], oracle_args)
    queue, prediction, probability, oracle_probability, history, trace = [], [], [], [], [], []
    truth_versions, inference_times, peak_state = {}, [], 0
    audit_counted = set()
    ground_truth = json.loads(Path(args.version_truth).read_text(encoding="utf-8")) if args.version_truth else {}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmup, len(x)):
        now = int(data["time"][index])
        start = time.perf_counter()
        p, cached = model.predict(x[index], tool[index], now)
        inference_times.append(time.perf_counter() - start)
        prediction.append(int(p.argmax()))
        probability.append(p)
        reference, ref_cache = oracle.predict(x[index], tool[index], now)
        oracle_probability.append(reference)
        oracle.feedback(x[index], y[index], tool[index], reference, ref_cache, now, now)
        heapq.heappush(queue, (now + int(data["delay"][index]), index, p, cached, now))
        while queue and queue[0][0] <= now:
            _, sample, issued_probability, cache, issued_time = heapq.heappop(queue)
            model.feedback(x[sample], y[sample], tool[sample], issued_probability, cache, issued_time, now)
        if args.inject_every and (index - args.warmup + 1) % args.inject_every == 0:
            model.inject(now)
        for identity, node in model.versions.items():
            if identity in audit_counted:
                continue
            audit_counted.add(identity)
            if str(identity) in ground_truth:
                truth_versions[identity] = bool(ground_truth[str(identity)])
            elif "audit_x" in data:
                phase = int(data["phase"][index])
                audit_x = (data["audit_x"][phase] - mean) / scale
                audit_y, audit_tool = data["audit_y"][phase], data["audit_tool"][phase]
                inside, outside = [], []
                for vector, label, machine in zip(audit_x, audit_y, audit_tool):
                    if machine != node.key[0]:
                        continue
                    before = model.probability(vector, machine, {node.key: node.before})
                    after = model.probability(vector, machine, {node.key: node.after})
                    truth = np.eye(model.classes)[int(label)]
                    difference = float(np.sum((after - truth) ** 2) - np.sum((before - truth) ** 2))
                    (inside if model.membership(vector)[node.key[1]] >= args.region_threshold else outside).append(difference)
                if inside:
                    truth_versions[identity] = bool(np.mean(inside) > args.harm_tolerance and (not outside or -np.mean(outside) <= args.outside_tolerance))
        if (index - args.warmup + 1) % args.batch_size == 0:
            unresolved = {record[0] for item in queue for record in item[3]}
            model.compact(now, unresolved)
            memory = len(pickle.dumps((model, queue), protocol=5))
            peak_state = max(peak_state, memory)
            history.append(dict(time=now, samples=len(prediction), macro_f1=float(f1_score(y[args.warmup:index + 1], prediction, average="macro", zero_division=0)),
                                rules=len(model.centers), active_versions=sum(node.status in ("committed", "canary") for node in model.versions.values()),
                                pending_labels=len(queue), retained_state_bytes=memory))
        trace.append(dict(index=index, time=now, prediction=prediction[-1], versions=[record[0] for record in cached]))
    probability, oracle_probability = np.asarray(probability), np.asarray(oracle_probability)
    target = y[args.warmup:]
    truth = np.eye(model.classes)[target]
    regret = np.sum((probability - truth) ** 2, 1) - np.sum((oracle_probability - truth) ** 2, 1)
    rollback = [event["version"] for event in model.events if event["action"] == "rollback"]
    audited = [identity for identity in rollback if identity in truth_versions]
    good = [identity for identity, harmful in truth_versions.items() if not harmful]
    result = dict(config=vars(args), macro_f1=float(f1_score(target, prediction, average="macro", zero_division=0)),
                  rare_fault_recall=float(recall_score(target, prediction, labels=[args.rare_class], average="macro", zero_division=0)),
                  delayed_regret=float(regret.sum()), rollback_precision=float(np.mean([truth_versions[identity] for identity in audited])) if audited else None,
                  false_rollback_rate=sum(identity in rollback for identity in good) / len(good) if good else None,
                  audited_rollbacks=len(audited), total_rollbacks=len(rollback), rule_count=len(model.centers),
                  active_versions=sum(node.status in ("committed", "canary") for node in model.versions.values()),
                  latency_ms=float(np.mean(inference_times) * 1000), peak_serialized_state_bytes=peak_state,
                  pending_feedback_at_end=len(queue), recovery=recovery(prediction, target, data["drift"], args.warmup, args.batch_size, args.recovery_tolerance))
    for name, value in (("metrics", result), ("history", history), ("versions", model.events), ("predictions", trace),
                        ("version_truth", truth_versions)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    np.savez_compressed(output / "probabilities.npz", probability=probability, oracle_probability=oracle_probability,
                        y=target, regret=regret, mean=mean, scale=scale)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    simulation = sub.add_parser("generate")
    simulation.add_argument("--output", required=True)
    simulation.add_argument("--samples", type=int, default=20000)
    simulation.add_argument("--features", type=int, default=8)
    simulation.add_argument("--tools", type=int, default=4)
    simulation.add_argument("--drift-interval", type=int, default=5000)
    simulation.add_argument("--rare-threshold", type=float, default=2)
    simulation.add_argument("--delay", type=int, default=100)
    simulation.add_argument("--delay-kind", choices=["fixed", "lognormal", "fault_dependent"], default="fixed")
    simulation.add_argument("--seed", type=int, default=13)
    data = sub.add_parser("prepare")
    data.add_argument("--csv", required=True)
    data.add_argument("--features", nargs="+", required=True)
    data.add_argument("--output", required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--data", required=True)
    execution.add_argument("--output", required=True)
    execution.add_argument("--version-truth")
    execution.add_argument("--variant", choices=["full", "no_rollback", "global_rollback", "no_canary", "no_compensation", "no_versions"], default="full")
    for name, default in (("warmup", 1000), ("rules", 8), ("minimum-feedback", 30), ("update-support", 10),
                          ("granule-support", 20), ("audit-horizon", 5000), ("cooldown", 100), ("batch-size", 100),
                          ("rare-class", 1), ("seed", 13), ("inject-every", 0)):
        execution.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("minimum-width", 0.5), ("canary-fraction", 0.2), ("confidence-count", 10),
                          ("novelty", 0.2), ("recency", 2000), ("forgetting", 0.95), ("alpha", 0.05),
                          ("commit-tolerance", 0.001), ("harm-tolerance", 0.01), ("outside-tolerance", 0.01),
                          ("region-threshold", 0.5), ("recovery-tolerance", 0.02)):
        execution.add_argument(f"--{name}", type=float, default=default)
    plotting = sub.add_parser("plot")
    plotting.add_argument("--runs", nargs="+", required=True)
    plotting.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "generate":
        if min(args.samples, args.features, args.tools, args.drift_interval) < 1 or args.delay < 0:
            parser.error("Positive stream sizes and nonnegative delay required")
        generate(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        if min(args.batch_size, args.update_support, args.minimum_feedback, args.recency, args.minimum_width) <= 0 or not 0 < args.canary_fraction <= 1 or not 0 < args.alpha < 1:
            parser.error("Invalid support, width, canary or confidence parameters")
        run(args)
    else:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        for metric in ("macro_f1", "rules", "active_versions", "pending_labels", "retained_state_bytes"):
            figure, axis = plt.subplots(figsize=(8, 4))
            for directory in args.runs:
                root = Path(directory)
                history = json.loads((root / "history.json").read_text(encoding="utf-8"))
                axis.plot([row["time"] for row in history], [row[metric] for row in history], label=root.name)
            axis.set(xlabel="Stream time", ylabel=metric)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / f"{metric}.png", dpi=180)
            plt.close(figure)
