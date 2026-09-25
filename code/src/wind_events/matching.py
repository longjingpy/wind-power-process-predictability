"""Deterministic one-to-one interval correspondence, with complete denominators.

Study definition: UTC/declared-source-clock interval IoU within a site, turbine,
split and hierarchy level. Assignment maximizes total feasible IoU in the
sensitivity arm (Kuhn, Naval Research Logistics Quarterly, 1955,
doi:10.1002/nav.3800020109). Zero-reward dummy columns permit unmatched events.
"""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def overlap_graph(left, right, cutoff=.5, measure='iou'):
    """Return all qualifying cross-protocol edges and connected components.

    Study-defined multi-interval comparison: time overlap is computed within
    a caller-supplied farm/unit/split. IoU is primary; overlap/min(duration)
    is a separate containment sensitivity for a leg nested in a V episode.
    Nonpositive intervals are invalid, not zero-score observations.
    """
    if measure not in ('iou','containment') or not 0 < cutoff <= 1:
        raise ValueError('Invalid overlap definition')
    left=sorted(left,key=lambda r:(r[1],r[2],r[0]))
    right=sorted(right,key=lambda r:(r[1],r[2],r[0]))
    for side in (left,right):
        if len({r[0] for r in side})!=len(side) or any(r[2]<=r[1] for r in side):
            raise ValueError('Unique IDs and positive intervals required')
    if not left or not right:return [],[]
    starts=np.array([r[1] for r in right],dtype=np.int64)
    longest=max(r[2]-r[1] for r in right)
    edges=[];parent={}
    def find(v):
        parent.setdefault(v,v)
        while parent[v]!=v:
            parent[v]=parent[parent[v]];v=parent[v]
        return v
    for aid,a0,a1 in left:
        lo=np.searchsorted(starts,a0-longest,side='right');hi=np.searchsorted(starts,a1,side='left')
        for bid,b0,b1 in right[lo:hi]:
            overlap=min(a1,b1)-max(a0,b0)
            if overlap<=0:continue
            union=(a1-a0)+(b1-b0)-overlap
            iou=overlap/union
            score=iou if measure=='iou' else overlap/min(a1-a0,b1-b0)
            if score<cutoff:continue
            a,b=(0,aid),(1,bid);ra,rb=find(a),find(b)
            if ra!=rb:parent[rb]=ra
            edges.append(dict(event_a=aid,event_b=bid,iou=float(iou),score=float(score),
                start=min(a0,b0),end=max(a1,b1)))
    groups={}
    for i,e in enumerate(edges):groups.setdefault(find((0,e['event_a'])),[]).append(i)
    components=[]
    for indices in groups.values():
        ee=[edges[i] for i in indices]
        components.append(dict(left_ids=sorted({e['event_a'] for e in ee}),right_ids=sorted({e['event_b'] for e in ee}),
            edge_indices=indices,start=min(e['start'] for e in ee),end=max(e['end'] for e in ee)))
    components.sort(key=lambda c:(c['start'],c['left_ids'][0],c['right_ids'][0]))
    return edges,components


def match_intervals(left, right, cutoff=.5, method="greedy"):
    """Inputs are (event_id, start_nanoseconds, end_nanoseconds) tuples."""
    if not 0 < cutoff <= 1 or method not in ("greedy", "maximum_iou"):
        raise ValueError("Invalid matching settings")
    left = sorted(left, key=lambda r: (r[1], r[2], r[0]))
    right = sorted(right, key=lambda r: (r[1], r[2], r[0]))
    for side in (left, right):
        if len({r[0] for r in side}) != len(side) or any(r[2] <= r[1] for r in side):
            raise ValueError("Unique IDs and positive intervals required")
    if not left or not right:
        return []
    starts = np.array([r[1] for r in right], np.int64)
    longest = max(r[2]-r[1] for r in right)
    edges = []
    for i, (aid, a0, a1) in enumerate(left):
        lo = np.searchsorted(starts, a0-longest, side="right")
        hi = np.searchsorted(starts, a1, side="left")
        for j in range(lo, hi):
            bid, b0, b1 = right[j]
            iou = (min(a1, b1)-max(a0, b0))/(max(a1, b1)-min(a0, b0))
            if iou >= cutoff:
                edges.append((i, j, float(iou), abs(a0-b0), aid, bid, min(a0, b0)))
    if method == "greedy":
        used_a, used_b, chosen = set(), set(), []
        for edge in sorted(edges, key=lambda e: (-e[2], e[3], e[4], e[5])):
            if edge[0] not in used_a and edge[1] not in used_b:
                used_a.add(edge[0]); used_b.add(edge[1]); chosen.append(edge)
    else:
        # Solve each connected overlap component; unrelated calendar periods
        # must not create a dense whole-archive assignment matrix.
        neighbours = {}
        for edge in edges:
            a, b = (0, edge[0]), (1, edge[1])
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)
        remaining, chosen = set(neighbours), []
        by_left = {}
        for edge in edges:
            by_left.setdefault(edge[0], []).append(edge)
        while remaining:
            todo, component = [min(remaining)], set()
            while todo:
                node = todo.pop()
                if node in component:
                    continue
                component.add(node); todo.extend(neighbours[node]-component)
            remaining -= component
            ai = sorted(n[1] for n in component if n[0] == 0)
            bi = sorted(n[1] for n in component if n[0] == 1)
            apos, bpos = {a: i for i, a in enumerate(ai)}, {b: i for i, b in enumerate(bi)}
            reward = np.full((len(ai), len(bi)+len(ai)), -1e6)
            reward[:, len(bi):] = 0
            lookup = {}
            for a in ai:
                for edge in by_left[a]:
                    key = (apos[a], bpos[edge[1]])
                    reward[key] = edge[2]; lookup[key] = edge
            rr, cc = linear_sum_assignment(reward, maximize=True)
            chosen.extend(lookup[r, c] for r, c in zip(rr, cc) if (r, c) in lookup)
    return [(e[4], e[5], e[2], e[6]) for e in chosen]


def pair_catalog(table, cutoff=.5, method="greedy", configurations=None):
    columns = ["site", "turbine", "split", "event_level"]
    output_columns = columns+["config_a", "config_b", "event_a", "event_b", "iou", "pair_time"]
    if not len(table):
        return pd.DataFrame(columns=output_columns), pd.DataFrame()
    if not table.event_id.is_unique:
        raise ValueError("Duplicate event IDs")
    data = table.copy()
    for field in ("time_start", "time_end"):
        data[field] = pd.to_datetime(data[field], utc=True).astype("datetime64[ns, UTC]").astype("int64")
    pairs, coverage = [], []
    for group, g in data.groupby(columns, sort=True):
        parts = {name: list(q[["event_id", "time_start", "time_end"]].itertuples(index=False, name=None))
                 for name, q in g.groupby("config", sort=True)}
        configs = sorted(configurations or parts)
        for ca, cb in combinations(configs, 2):
            left, right = parts.get(ca, []), parts.get(cb, [])
            matches = match_intervals(left, right, cutoff, method)
            identity = dict(zip(columns, group)) | {"config_a": ca, "config_b": cb}
            for a, b, iou, stamp in matches:
                pairs.append(identity | {"event_a": a, "event_b": b, "iou": iou,
                                        "pair_time": pd.Timestamp(stamp, tz="UTC")})
            coverage.append(identity | {"left_n": len(left), "right_n": len(right), "pairs": len(matches),
                            "left_coverage": len(matches)/len(left) if left else None,
                            "right_coverage": len(matches)/len(right) if right else None,
                            "cutoff": cutoff, "matching": method})
    return pd.DataFrame(pairs, columns=output_columns), pd.DataFrame(coverage)
