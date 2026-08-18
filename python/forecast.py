#!/usr/bin/env python3
"""
Diesel Control — forecast and static data builder
Panda Shoes Industry Ltd

Pulls the daily fuel ledger, forecasts consumption, projects the tank against
open reservations, and writes JSON that GitHub Pages serves to the dashboard.
Runs on the standard library only — no pip install needed in CI.

Usage
    python forecast.py --api  https://script.google.com/.../exec
    python forecast.py --csv  data/daily.csv
    python forecast.py --sample                 # generate demo CSVs to work against
    python forecast.py --api ... --horizon 45 --quiet

Outputs (into ../data/)
    daily.json      the ledger the dashboard reads when the API is unreachable
    forecast.json   forecast rows, stock projection, order recommendation
    summary.json    the bird's-eye numbers, for email or Slack

The forecast method matches the one in index.html, so the browser and the
scheduled job never disagree about the numbers.
"""

import argparse
import csv
import json
import math
import os
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

DEFAULT_CONFIG = {
    "tankCapacity": 20000,
    "dayTank": 2000,
    "safetyStock": 5000,
    "leadTimeDays": 3,
    "targetCoverDays": 21,
    "orderMultiple": 1000,
    "minOrder": 3000,
}

ALPHA, BETA = 0.22, 0.06     # Holt smoothing — same constants as the dashboard
Z80 = 1.28                   # 80% prediction interval


# --------------------------------------------------------------- loading
def load_api(url):
    """Read the Apps Script web app feed."""
    with urllib.request.urlopen(url + "?action=data", timeout=60) as r:
        payload = json.loads(r.read().decode("utf-8"))
    days = payload.get("days", [])
    cfg = dict(DEFAULT_CONFIG, **{k: v for k, v in (payload.get("config") or {}).items()
                                  if isinstance(v, (int, float))})
    return days, payload.get("reservations", []), cfg


def load_csv(path):
    """Read a flat daily ledger: date,opening,received,consumed,closing,dip,pairs,rate,cost"""
    days = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_equip = {k[3:]: float(v or 0) for k, v in row.items()
                        if k.startswith("eq_") and v not in (None, "")}
            days.append({
                "byEquip": by_equip,
                "date": row["date"][:10],
                "opening": float(row.get("opening") or 0),
                "received": float(row.get("received") or 0),
                "consumed": float(row.get("consumed") or 0),
                "closing": float(row.get("closing") or 0),
                "dip": float(row.get("dip") or 0),
                "pairs": float(row.get("pairs") or 0),
                "rate": float(row.get("rate") or 0),
                "cost": float(row.get("cost") or 0),
            })
    days.sort(key=lambda d: d["date"])
    res_path = os.path.join(os.path.dirname(path), "reservations.csv")
    reservations = []
    if os.path.exists(res_path):
        with open(res_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status", "").lower() in ("confirmed", "in-transit"):
                    reservations.append({
                        "po": row["po"], "supplier": row.get("supplier", ""),
                        "qty": float(row["qty"]), "rate": float(row.get("rate") or 0),
                        "arrive": row["expected"][:10], "status": row["status"].lower(),
                    })
    return days, reservations, dict(DEFAULT_CONFIG)


# --------------------------------------------------------------- forecast
def weekday_profile(days, window=84):
    """Multiplicative day-of-week factors. Friday is the weekly holiday here,
    so the profile matters more than it would in a 7-day operation."""
    tail = days[-window:]
    vals = [d["consumed"] for d in tail]
    mean = sum(vals) / len(vals) if vals else 1.0
    tot = [0.0] * 7
    cnt = [0] * 7
    for d in tail:
        w = datetime.strptime(d["date"], "%Y-%m-%d").weekday()   # Mon=0 … Sun=6
        tot[w] += d["consumed"] / (mean or 1)
        cnt[w] += 1
    prof = [(tot[i] / cnt[i]) if cnt[i] else 1.0 for i in range(7)]
    norm = (sum(prof) / 7) or 1.0
    return [min(2.2, max(0.15, p / norm)) for p in prof]


def holt(days, horizon, prof):
    """Holt linear trend on deseasonalised demand, reseasonalised on the way out."""
    des, dates = [], []
    for d in days:
        w = datetime.strptime(d["date"], "%Y-%m-%d").weekday()
        des.append(d["consumed"] / prof[w])
        dates.append(d["date"])
    if len(des) < 21:
        raise SystemExit("Need at least 21 days of history to forecast.")

    level = sum(des[:14]) / 14
    trend = (sum(des[7:14]) - sum(des[:7])) / 49
    errs = []
    for i in range(14, len(des)):
        errs.append(des[i] - (level + trend))
        prev = level
        level = ALPHA * des[i] + (1 - ALPHA) * (level + trend)
        trend = BETA * (level - prev) + (1 - BETA) * trend

    recent = errs[-120:] or errs
    sd = math.sqrt(sum(e * e for e in recent) / len(recent))

    t0 = datetime.strptime(dates[-1], "%Y-%m-%d").date()
    rows = []
    for h in range(1, horizon + 1):
        d = t0 + timedelta(days=h)
        f = prof[d.weekday()]
        mean = max(0.0, level + trend * h) * f
        band = Z80 * sd * math.sqrt(h) * f
        rows.append({"date": d.isoformat(),
                     "mean": round(mean, 1),
                     "lo": round(max(0.0, mean - band), 1),
                     "hi": round(mean + band, 1)})
    return rows, sd, level, trend


def project(rows, reservations, stock, cfg):
    """Walk the tank forward with and without the volume already on order."""
    sched = {}
    for r in reservations:
        sched[r["arrive"]] = sched.get(r["arrive"], 0) + r["qty"]
    bare = with_res = stock
    out = []
    for r in rows:
        bare -= r["mean"]
        with_res += sched.get(r["date"], 0) - r["mean"]
        out.append({"date": r["date"], "bare": round(bare, 1),
                    "withRes": round(with_res, 1), "recv": sched.get(r["date"], 0)})
    first = lambda key, lim: next((o["date"] for o in out if o[key] < lim), None)
    return {
        "rows": out,
        "dry": first("bare", 0),
        "safetyHit": first("bare", cfg["safetyStock"]),
        "dryWithReservations": first("withRes", 0),
        "safetyHitWithReservations": first("withRes", cfg["safetyStock"]),
    }


def recommend(stock, on_order, burn7, cfg, proj):
    target = burn7 * cfg["targetCoverDays"] + cfg["safetyStock"]
    need = max(0.0, target - stock - on_order)
    mult = cfg["orderMultiple"]
    qty = math.ceil(need / mult) * mult if need else 0
    ullage = math.floor(max(0.0, cfg["tankCapacity"] - stock - on_order) / mult) * mult
    qty = int(max(0, min(qty, ullage)))          # never order more than will fit
    ref = proj["safetyHitWithReservations"] or proj["safetyHit"]
    order_by = None
    if ref:
        order_by = (datetime.strptime(ref, "%Y-%m-%d").date()
                    - timedelta(days=int(cfg["leadTimeDays"]))).isoformat()
    return {
        "quantity": qty if qty >= cfg["minOrder"] else 0,
        "orderBy": order_by,
        "reorderPoint": round(burn7 * cfg["leadTimeDays"] + cfg["safetyStock"]),
        "targetCoverDays": cfg["targetCoverDays"],
    }


# --------------------------------------------------------------- periods
def periods(days, mode):
    def key(d):
        y, m = d[:4], int(d[5:7])
        return {"daily": d, "monthly": d[:7],
                "quarterly": "%s-Q%d" % (y, (m + 2) // 3), "yearly": y}[mode]

    agg = {}
    for d in days:
        k = key(d["date"])
        a = agg.setdefault(k, {"key": k, "days": 0, "consumed": 0.0, "received": 0.0,
                               "cost": 0.0, "pairs": 0.0, "closing": 0.0})
        a["days"] += 1
        for f in ("consumed", "received", "cost", "pairs"):
            a[f] += d.get(f, 0) or 0
        a["closing"] = d["closing"]

    rows = [agg[k] for k in sorted(agg)]
    back = {"daily": 7, "monthly": 12, "quarterly": 4, "yearly": 1}[mode]
    for i, r in enumerate(rows):
        r["perDay"] = round(r["consumed"] / r["days"], 1)
        r["rate"] = round(r["cost"] / r["received"], 2) if r["received"] else 0
        r["litresPer1000Pairs"] = round(r["consumed"] / r["pairs"] * 1000, 1) if r["pairs"] else 0
        r["expectedDays"] = expected_days(r["key"], mode)
        r["partial"] = r["days"] < r["expectedDays"]
        # deltas run on litres per day, never on totals: a part-month or a
        # 28-day February would otherwise read as a fall in consumption.
        prev = rows[i - 1]["perDay"] if i else 0
        r["deltaPrevPct"] = round((r["perDay"] - prev) / prev * 100, 1) if prev else None
        r["deltaYoYPct"] = (round((r["perDay"] - rows[i - back]["perDay"])
                                  / rows[i - back]["perDay"] * 100, 1)
                            if i >= back and rows[i - back]["perDay"] else None)
        for f in ("consumed", "received", "cost", "pairs"):
            r[f] = round(r[f])
    return rows


def expected_days(key, mode):
    """How many days a complete period of this type holds."""
    if mode == "daily":
        return 1
    if mode == "monthly":
        y, m = int(key[:4]), int(key[5:7])
        nxt = date(y + (m == 12), 1 if m == 12 else m + 1, 1)
        return (nxt - date(y, m, 1)).days
    if mode == "quarterly":
        y, q = int(key[:4]), int(key[-1])
        start = date(y, 3 * q - 2, 1)
        end = date(y + (q == 4), 1 if q == 4 else 3 * q + 1, 1)
        return (end - start).days
    y = int(key)
    return (date(y + 1, 1, 1) - date(y, 1, 1)).days


# --------------------------------------------------------------- report
def bird_eye(days, reservations, cfg, fc_rows, proj, rec):
    last = days[-1]
    burn = lambda k: sum(d["consumed"] for d in days[-k:]) / min(k, len(days))
    burn7, burn30 = burn(7), burn(30)
    on_order = sum(r["qty"] for r in reservations)
    month = last["date"][:7]
    mtd = [d for d in days if d["date"][:7] == month]
    tot = lambda arr, f: sum(d[f] for d in arr)
    f30 = sum(r["mean"] for r in fc_rows)
    mtd_recv = tot(mtd, "received")
    return {
        "asOf": last["date"],
        "stock": round(last["closing"]),
        "dip": round(last.get("dip") or last["closing"]),
        "dipVariance": round((last.get("dip") or last["closing"]) - last["closing"]),
        "onOrder": round(on_order),
        "openOrders": len(reservations),
        "burn7": round(burn7),
        "burn30": round(burn30),
        "coverDays": round(last["closing"] / burn7, 1) if burn7 else None,
        "coverDaysWithReservations": round((last["closing"] + on_order) / burn7, 1) if burn7 else None,
        "reorderPoint": rec["reorderPoint"],
        "safetyStockHit": proj["safetyHitWithReservations"],
        "dryTank": proj["dryWithReservations"],
        "mtdConsumed": round(tot(mtd, "consumed")),
        "mtdReceived": round(mtd_recv),
        "mtdCost": round(tot(mtd, "cost")),
        "avgRate": round(tot(mtd, "cost") / mtd_recv, 2) if mtd_recv else None,
        "forecast30": round(f30),
        "forecast30Cost": round(f30 * (tot(mtd, "cost") / mtd_recv)) if mtd_recv else None,
        "suggestedOrder": rec["quantity"],
        "orderBy": rec["orderBy"],
    }


def print_report(s):
    line = lambda k, v: print("  %-32s %s" % (k, v))
    print("\n" + "=" * 58)
    print("  DIESEL CONTROL — Panda Shoes Industry Ltd")
    print("  Bird's-eye summary as of %s" % s["asOf"])
    print("=" * 58)
    line("In tank (book)", "%s L" % f"{s['stock']:,}")
    line("Physical dip", "%s L  (variance %+d L)" % (f"{s['dip']:,}", s["dipVariance"]))
    line("Reserved on order", "%s L across %d order(s)" % (f"{s['onOrder']:,}", s["openOrders"]))
    line("Burn 7 / 30 day", "%s / %s L per day" % (f"{s['burn7']:,}", f"{s['burn30']:,}"))
    line("Days of cover", "%s  (%s with reservations)" % (s["coverDays"], s["coverDaysWithReservations"]))
    line("Reorder point", "%s L" % f"{s['reorderPoint']:,}")
    line("Safety stock reached", s["safetyStockHit"] or "not within horizon")
    line("Month to date", "%s L consumed · %s L received" % (f"{s['mtdConsumed']:,}", f"{s['mtdReceived']:,}"))
    line("Month to date purchase", "BDT %s" % f"{s['mtdCost']:,}")
    line("Forecast next 30 days", "%s L" % f"{s['forecast30']:,}")
    print("-" * 58)
    if s["suggestedOrder"]:
        print("  ACTION  Order %s L by %s" % (f"{s['suggestedOrder']:,}", s["orderBy"] or "this week"))
    else:
        print("  ACTION  No purchase needed — cover is at policy target.")
    print("=" * 58 + "\n")


# --------------------------------------------------------------- sample data
SAMPLE_EQUIPMENT = [          # id, litres per working day, load-shedding sensitive
    ("DG-01", 415, True),
    ("DG-02", 88, True),
    ("BLR-01", 148, False),
    ("FLT", 37, False),
    ("VEH", 54, False),
]


def write_sample():
    """Deterministic sample ledger so the pipeline can be tested end to end."""
    import random
    rnd = random.Random(818202601)
    os.makedirs(DATA, exist_ok=True)
    start, today = date(2024, 1, 1), date.today()
    stock, rows, res, po = 13800.0, [], [], 1
    pending = []
    d = start
    while d <= today:
        friday = d.weekday() == 4
        season = 1 + 0.34 * math.sin((d.month - 3) / 12 * 2 * math.pi)
        run = 0.22 if friday else 0.94 + rnd.random() * 0.14
        split = {}
        for eq_id, base, seasonal in SAMPLE_EQUIPMENT:
            q = base * run * (0.9 + rnd.random() * 0.2)
            if seasonal:
                q *= 0.85 + 0.3 * season
            split[eq_id] = round(q)
        split["FP-01"] = 14 if d.weekday() == 5 else 0     # weekly fire-pump test
        consumed = sum(split.values())
        pairs = round((2100 if friday else 10400) * (0.92 + rnd.random() * 0.18))
        received = 0.0
        keep = []
        for p in pending:
            if p["arrive"] == d.isoformat():
                received += p["qty"]
            else:
                keep.append(p)
        pending = keep
        rate = round(104 + 4 * math.sin((d - start).days / 220) + rnd.uniform(-0.3, 0.3), 2)
        opening = stock
        stock = opening + received - consumed
        row = {"date": d.isoformat(), "opening": round(opening), "received": round(received),
               "consumed": consumed, "closing": round(stock),
               "dip": round(stock * (1 + rnd.uniform(-0.008, 0.008))),
               "pairs": pairs, "rate": rate, "cost": round(received * rate)}
        row.update({"eq_" + k: v for k, v in split.items()})
        rows.append(row)
        burn = sum(r["consumed"] for r in rows[-7:]) / min(7, len(rows))
        on_order = sum(p["qty"] for p in pending)
        if stock + on_order < burn * 3 + 5000:
            qty = math.ceil((burn * 21 + 5000 - stock - on_order) / 1000) * 1000
            qty = max(3000, min(qty, 20000 - round(stock) - on_order))
            if qty >= 3000:
                p = {"po": "PSIL/DSL/%d/%03d" % (d.year, po), "supplier": "Padma Oil Co. Ltd",
                     "qty": qty, "rate": rate, "ordered": d.isoformat(),
                     "arrive": (d + timedelta(days=3)).isoformat(), "status": "confirmed"}
                po += 1
                pending.append(p)
                res.append(p)
        d += timedelta(days=1)

    # leave one forward-dated order open so the reservation pipeline is visible
    pending.append({"po": "PSIL/DSL/%d/%03d" % (today.year, po),
                    "supplier": "Meghna Petroleum Ltd", "qty": 8000, "rate": rows[-1]["rate"],
                    "ordered": today.isoformat(),
                    "arrive": (today + timedelta(days=7)).isoformat(), "status": "confirmed"})

    with open(os.path.join(DATA, "daily.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(DATA, "reservations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["po", "ordered", "supplier", "qty", "rate", "expected", "status"])
        for p in pending:
            w.writerow([p["po"], p["ordered"], p["supplier"], p["qty"], p["rate"], p["arrive"], "confirmed"])
    print("Wrote sample ledger: %d days → %s" % (len(rows), DATA))


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Diesel forecast for Panda Shoes Industry Ltd")
    ap.add_argument("--api", help="Apps Script web app /exec URL")
    ap.add_argument("--csv", help="local daily ledger CSV")
    ap.add_argument("--horizon", type=int, default=30, help="forecast days (default 30)")
    ap.add_argument("--sample", action="store_true", help="write a sample ledger and exit")
    ap.add_argument("--quiet", action="store_true", help="suppress the printed report")
    args = ap.parse_args()

    if args.sample:
        write_sample()
        return

    if args.api:
        days, reservations, cfg = load_api(args.api)
    elif args.csv:
        days, reservations, cfg = load_csv(args.csv)
    else:
        default = os.path.join(DATA, "daily.csv")
        if not os.path.exists(default):
            ap.error("give --api or --csv (or run --sample first)")
        days, reservations, cfg = load_csv(default)

    if not days:
        sys.exit("No ledger rows found.")

    prof = weekday_profile(days)
    rows, sd, level, trend = holt(days, args.horizon, prof)
    stock = days[-1]["closing"]
    on_order = sum(r["qty"] for r in reservations)
    burn7 = sum(d["consumed"] for d in days[-7:]) / min(7, len(days))
    proj = project(rows, reservations, stock, cfg)
    rec = recommend(stock, on_order, burn7, cfg, proj)
    summary = bird_eye(days, reservations, cfg, rows, proj, rec)

    os.makedirs(DATA, exist_ok=True)
    write = lambda name, obj: json.dump(
        obj, open(os.path.join(DATA, name), "w", encoding="utf-8"), separators=(",", ":"))

    write("daily.json", {"days": days, "reservations": reservations,
                         "config": cfg, "source": "python",
                         "generated": datetime.now(timezone.utc).isoformat()})
    write("forecast.json", {
        "generated": datetime.now(timezone.utc).isoformat(),
        "method": "Holt(alpha=%.2f, beta=%.2f) x weekday profile" % (ALPHA, BETA),
        "residualSd": round(sd, 1), "level": round(level, 1), "trendPerDay": round(trend, 2),
        "weekdayProfile": dict(zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                   [round(p, 3) for p in prof])),
        "rows": rows, "projection": proj, "recommendation": rec,
    })
    write("summary.json", summary)
    write("periods.json", {m: (periods(days, m)[-90:] if m == "daily" else periods(days, m))
                           for m in ("daily", "monthly", "quarterly", "yearly")})

    if not args.quiet:
        print_report(summary)
        print("  Weekday profile: " + "  ".join(
            "%s %.2f" % (n, p) for n, p in zip(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], prof)))
        print("  Written to %s\n" % DATA)


if __name__ == "__main__":
    main()
