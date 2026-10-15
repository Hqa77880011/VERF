from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans


@dataclass
class Version:
    identity: int
    parent: int
    key: tuple
    before: np.ndarray
    after: np.ndarray
    created: int
    status: str = "canary"
    injected: bool = False
    observations: list = field(default_factory=list)
    error_counts: np.ndarray | None = None
    exposure: int = 0


class VersionedFuzzy:
    def __init__(self, x, y, tool, args):
        self.args = args
        self.rng = np.random.default_rng(args.seed)
        self.classes = int(max(y)) + 1
        if len(np.unique(tool)) < 2:
            raise ValueError("Shared granules require observations from at least two tools")
        self.prior = np.bincount(y, minlength=self.classes).astype(float) + 1
        clusters = KMeans(n_clusters=min(args.rules, len(x)), random_state=args.seed, n_init=10).fit(x)
        self.centers, self.widths, self.counts = [], [], {}
        for index in range(clusters.n_clusters):
            selected = clusters.labels_ == index
            if len(np.unique(tool[selected])) < min(2, len(np.unique(tool))):
                continue
            self.centers.append(clusters.cluster_centers_[index])
            self.widths.append(np.maximum(x[selected].std(0), args.minimum_width))
        if not self.centers:
            raise ValueError("Warmup needs a compatible shared region observed on two tools")
        self.centers, self.widths = list(self.centers), list(self.widths)
        for vector, label, machine in zip(x, y, tool):
            membership = self.membership(vector)
            rule = int(membership.argmax())
            key = (int(machine), rule)
            self.counts.setdefault(key, np.zeros(self.classes))[int(label)] += 1
        self.active, self.versions, self.pending = {}, {}, {}
        self.latest, self.next_identity = -1, 0
        self.cooldown, self.reversals = {}, {}
        self.events, self.novel = [], []
        self.drift = {}

    def membership(self, x):
        center, width = np.asarray(self.centers), np.asarray(self.widths)
        distance = ((x - center) / width) ** 2
        lower = np.exp(-0.5 * np.mean(distance * 1.25 ** 2, 1))
        upper = np.exp(-0.5 * np.mean(distance / 1.25 ** 2, 1))
        return (lower + upper) / 2

    def key_counts(self, key, replacements=None):
        if replacements is not None and key in replacements:
            return replacements[key]
        return self.counts.get(key, np.zeros(self.classes))

    def probability(self, x, tool, replacements=None):
        membership = self.membership(x)
        adjusted = membership / (1 + self.drift.get(int(tool), 0))
        mass, weighted = 0.0, np.zeros(self.classes)
        for rule, activation in enumerate(adjusted):
            counts = self.key_counts((int(tool), rule), replacements)
            support = counts.sum()
            if support:
                weight = activation * support / (support + self.args.confidence_count)
                weighted += weight * (counts + 1) / (support + self.classes)
                mass += weight
        return weighted / mass if mass > 1e-12 else self.prior / self.prior.sum()

    def predict(self, x, tool, now):
        replacements = {}
        used = []
        for key, identity in self.active.items():
            version = self.versions[identity]
            if key[0] != int(tool) or version.status != "canary":
                continue
            membership = self.membership(x)[key[1]]
            agreement = 1 / (1 + self.drift.get(int(tool), 0))
            fraction = 1 if self.args.variant == "no_canary" else self.args.canary_fraction * membership * agreement
            if self.rng.random() >= fraction:
                replacements[key] = version.before
            else:
                version.exposure += 1
                used.append(identity)
        probability = self.probability(x, tool, replacements)
        cache = []
        for key, identity in self.active.items():
            version = self.versions[identity]
            if key[0] != int(tool) or key in replacements or version.status not in ("committed", "canary"):
                continue
            parent_replacements = dict(replacements)
            parent_replacements[key] = version.before
            counterfactual = self.probability(x, tool, parent_replacements)
            cache.append((identity, counterfactual, float(self.membership(x)[key[1]])))
        return probability, cache

    def propose(self, key, after, now, injected=False, status="canary"):
        if self.args.variant == "no_versions":
            self.counts[key] = after.copy()
            return -1
        previous = self.active.get(key)
        if previous is not None and self.versions[previous].status == "canary":
            return None
        if now < self.cooldown.get(key, -1):
            return None
        before = self.key_counts(key).copy()
        identity = self.next_identity
        self.next_identity += 1
        version = Version(identity, self.latest, key, before, after.copy(), now, status, injected,
                          error_counts=np.zeros(self.classes))
        self.versions[identity] = version
        self.active[key] = identity
        self.counts[key] = after.copy()
        self.latest = identity
        self.events.append(dict(time=now, action="propose", version=identity, parent=version.parent,
                                tool=key[0], rule=key[1], injected=injected, status=status))
        return identity

    def descendants(self, identity):
        found, frontier = [], [identity]
        while frontier:
            parent = frontier.pop()
            children = [version.identity for version in self.versions.values() if version.parent == parent]
            found.extend(children)
            frontier.extend(children)
        return found

    def rollback(self, identity, now):
        version = self.versions[identity]
        descendants = self.descendants(identity)
        dependent = [self.versions[index] for index in descendants if self.versions[index].key == version.key
                     and self.versions[index].status in ("committed", "canary")]
        if self.args.variant == "global_rollback":
            for index in sorted(descendants, reverse=True):
                node = self.versions[index]
                self.counts[node.key] = node.before.copy()
                node.status = "rolled_back"
            self.counts[version.key] = version.before.copy()
        elif dependent:
            if self.args.variant == "no_compensation":
                version.status = "unresolved_harmful"
                self.events.append(dict(time=now, action="unresolved_harmful", version=identity))
                return
            current = self.active[version.key]
            self.versions[current].status = "superseded"
            correction = version.before + version.error_counts
            version.status = "rolled_back"
            self.propose(version.key, correction, now)
        else:
            self.counts[version.key] = version.before.copy()
            for index in descendants:
                node = self.versions[index]
                if node.parent == identity:
                    node.parent = version.parent
            if self.active.get(version.key) == identity:
                self.active.pop(version.key)
        version.status = "rolled_back"
        self.reversals[version.key] = self.reversals.get(version.key, 0) + 1
        self.cooldown[version.key] = now + self.args.cooldown
        self.events.append(dict(time=now, action="rollback", version=identity, injected=version.injected,
                                dependent_descendants=len(dependent)))

    def feedback(self, x, y, tool, predicted, cache, issued, now):
        truth = np.eye(self.classes)[int(y)]
        for identity, parent_prediction, membership in cache:
            if identity not in self.versions:
                continue
            version = self.versions[identity]
            if version.status not in ("canary", "committed"):
                continue
            weight = membership * np.exp(-(now - issued) / self.args.recency)
            difference = float(np.sum((predicted - truth) ** 2) - np.sum((parent_prediction - truth) ** 2))
            version.observations.append((issued, weight, difference))
            version.observations = [row for row in version.observations if now - row[0] <= self.args.audit_horizon]
            version.error_counts[int(y)] += membership
            weights = np.array([row[1] for row in version.observations])
            changes = np.array([row[2] for row in version.observations])
            effective = weights.sum() ** 2 / (np.sum(weights ** 2) + 1e-12)
            minimum = self.args.minimum_feedback * (1 + self.reversals.get(version.key, 0))
            if effective < minimum:
                continue
            average = float(np.sum(weights * changes) / weights.sum())
            variance = float(np.sum(weights * (changes - average) ** 2) / weights.sum())
            radius = norm.ppf(1 - self.args.alpha / 2) * np.sqrt(variance / effective + 1e-12)
            if average - radius > self.args.harm_tolerance and self.args.variant not in ("no_rollback", "no_versions"):
                self.rollback(identity, now)
            elif average + radius < self.args.commit_tolerance and version.status == "canary":
                version.status = "committed"
                self.events.append(dict(time=now, action="commit", version=identity, effective_support=float(effective),
                                        mean_loss_change=average, lower=average - radius, upper=average + radius))
        membership = self.membership(x)
        self.drift[int(tool)] = 0.95 * self.drift.get(int(tool), 0) + 0.05 * float(1 - membership.max())
        rule = int(membership.argmax())
        if membership[rule] < self.args.novelty:
            self.novel.append((x.copy(), int(y), int(tool), now))
            self.novel = [row for row in self.novel if now - row[3] <= self.args.audit_horizon]
            close = [row for row in self.novel if np.mean((row[0] - x) ** 2) <= self.args.minimum_width ** 2 * 4]
            if len(close) >= self.args.granule_support and len(set(row[2] for row in close)) >= 2:
                vectors = np.stack([row[0] for row in close])
                self.centers.append(vectors.mean(0))
                self.widths.append(vectors.std(0).clip(self.args.minimum_width))
                rule = len(self.centers) - 1
                self.events.append(dict(time=now, action="shared_granule", rule=rule, tools=sorted(set(row[2] for row in close))))
                self.novel = [row for row in self.novel if np.mean((row[0] - x) ** 2) > self.args.minimum_width ** 2 * 4]
        key = (int(tool), rule)
        self.pending.setdefault(key, np.zeros(self.classes))[int(y)] += 1
        if self.pending[key].sum() >= self.args.update_support:
            after = self.args.forgetting * self.key_counts(key) + self.pending[key]
            status = "committed" if self.args.variant == "no_versions" else "canary"
            identity = self.propose(key, after, now, status=status)
            if identity is not None:
                self.pending[key] = np.zeros(self.classes)

    def inject(self, now):
        for key in sorted(self.counts):
            previous = self.active.get(key)
            if previous is None or self.versions[previous].status != "canary":
                self.propose(key, np.roll(self.key_counts(key), 1) + 1, now, injected=True)
                return True
        return False

    def compact(self, now, unresolved_ids):
        for identity in list(self.versions):
            node = self.versions[identity]
            if now - node.created <= self.args.audit_horizon or identity in unresolved_ids or identity in self.active.values():
                continue
            children = [child for child in self.versions.values() if child.parent == identity]
            if node.status == "committed" and len(children) == 1 and children[0].status == "committed":
                children[0].parent = node.parent
                del self.versions[identity]
