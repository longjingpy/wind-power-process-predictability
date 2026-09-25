"""Minimal runnable interface; study-scale schedules live in research scripts."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from .protocol import Protocol, regularize
from .catalog import build_catalog
from .matching import pair_catalog
from .representation import Representation, fit_cluster
from .metrics import contingency, scores_from_counts


def main():
    parser = argparse.ArgumentParser(prog="wind-events")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="Run a synthetic pipeline verification")
    demo.add_argument("--output", type=Path, required=True)
    detect = sub.add_parser("catalog", help="Build a catalogue from timestamp,power,valid CSV")
    detect.add_argument("input", type=Path)
    detect.add_argument("--output", type=Path, required=True)
    detect.add_argument("--native-minutes", type=int, required=True)
    detect.add_argument("--resolution-minutes", type=int, default=30)
    detect.add_argument("--site", required=True)
    detect.add_argument("--turbine", required=True)
    detect.add_argument("--timestamp-position", choices=["start", "end", "unverified"], default="unverified")
    args = parser.parse_args()
    if args.command == "demo":
        times = pd.date_range("2020-01-01", periods=2000, freq="30min", tz="UTC")
        phase = np.arange(len(times))
        power = .5+.3*np.sin(phase/8)+.1*np.sin(phase/2)
        valid = np.ones(len(times), bool)
        valid[300:320] = False
        site, turbine, protocol, clock = "synthetic", "T1", Protocol(), "SYNTHETIC_CLOCK"
    else:
        data = pd.read_csv(args.input)
        if not {"timestamp", "power", "valid"}.issubset(data):
            raise ValueError("Required CSV columns: timestamp, power, valid")
        valid = data.valid.map({True: True, False: False, "True": True, "False": False, "true": True, "false": False, 1: True, 0: False})
        if valid.isna().any():
            raise ValueError("Validity must be explicit booleans")
        times, power, valid, _ = regularize(data.timestamp, data.power, valid, args.native_minutes,
                                           args.resolution_minutes, args.timestamp_position)
        site, turbine = args.site, args.turbine
        protocol, clock = Protocol(resolution_minutes=args.resolution_minutes), "USER_DECLARED_SOURCE_CLOCK"
    events, shapes, audit = build_catalog(site, turbine, times, power, valid, protocol,
                                         calibration="training_q995", clock_status=clock)
    args.output.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.output/"events.csv.gz", index=False)
    np.save(args.output/"shapes.npy", shapes)
    pairs, coverage = pair_catalog(events[events.representation_eligible & events.split.eq("test")])
    pairs.to_csv(args.output/"pairs.csv.gz", index=False)
    coverage.to_csv(args.output/"coverage.csv", index=False)
    audit["test_pairs"] = len(pairs)
    if args.command == "demo":
        training_rows = events.loc[events.representation_eligible & events.split.eq("train"), "shape_row"].to_numpy(int)
        transform = Representation("raw25").fit(shapes[training_rows])
        cluster = fit_cluster(transform.transform(shapes[training_rows]), "kmeans", 4, 41)
        labels = cluster.predict(transform.transform(shapes))
        lookup = events.set_index("event_id").shape_row
        left, right = lookup.loc[pairs.event_a].to_numpy(int), lookup.loc[pairs.event_b].to_numpy(int)
        if (left < 0).any() or (right < 0).any():
            raise IndexError("Matched demo events require valid shapes")
        nmi, ari, agreement, informative = scores_from_counts(contingency(labels[left], labels[right], 4))
        audit["synthetic_smoke_metrics"] = {"nmi": float(nmi[0]), "ari": float(ari[0]),
            "exact_agreement": float(agreement[0]), "informative": bool(informative[0]), "training_events": len(training_rows)}
    (args.output/"audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps({"events": len(events), "shapes": len(shapes), "test_pairs": len(pairs),
                      "synthetic_smoke_metrics": audit.get("synthetic_smoke_metrics")}, indent=2))


if __name__ == "__main__":
    main()
