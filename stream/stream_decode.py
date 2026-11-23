#!/usr/bin/env python
# Real-time CEBRA: replays the held-out rat data one 25 ms sample at a time,
# as if it were coming off the rat live, and drives two RGB LEDs on an ESP32.
#   LED 1 = where the rat really is
#   LED 2 = where CEBRA thinks it is, from the neural data
#
#   python stream/stream_decode.py --port /dev/ttyUSB0 --ack
#   python stream/stream_decode.py --model models/cebra_posdir3_shuffled_split.pt --control --port /dev/ttyUSB0 --ack
#   python stream/stream_decode.py --preview                  # no hardware, colors in the terminal
import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BIN_S, ROOT, ColorMap, Decoder, WindowEncoder, ansi_block,  # noqa: E402
                    load_data, load_model, shuffled_labels)


def main():
    ap = argparse.ArgumentParser(description="Real-time CEBRA position decoding with two RGB LEDs")
    ap.add_argument("--model", default=str(ROOT / "models/cebra_posdir3_split.pt"))
    ap.add_argument("--control", action="store_true",
                    help="fit the decoder on shuffled labels (use with the shuffled model)")
    ap.add_argument("--port", help="ESP32 serial port, e.g. /dev/ttyUSB0. Leave out to run with no hardware")
    ap.add_argument("--ack", action="store_true", help="wait for the ESP32 to confirm each packet")
    ap.add_argument("--preview", action="store_true", help="show both LED colors in the terminal")
    ap.add_argument("--steps", type=int, help="only stream this many samples")
    ap.add_argument("--out", help="results folder (default results/<model>_<time>)")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)  # one sample at a time is fastest on a single thread

    name = Path(args.model).stem + ("_control" if args.control else "") + time.strftime("_%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)

    # ---- setup (not timed) --------------------------------------------------------------
    print("loading data and model...")
    neural, label, split = load_data()
    enc = WindowEncoder(load_model(args.model))
    delay = enc.right - 1  # the window needs this many future samples, so each decode is this late

    # fit the kNN decoder on the training embeddings, same as the offline demo
    emb_train = enc.embed_all(neural[:split])
    decoder = Decoder(emb_train, shuffled_labels(label)[:split] if args.control else label[:split])
    colors = ColorMap(label[:split, 0])

    link = None
    if args.port:
        from led_link import LedLink
        link = LedLink(args.port, ack=args.ack)
        print(f"connected to ESP32 on {args.port}")

    # warm up (first torch / sklearn calls are slow)
    w = enc.width
    for _ in range(20):
        decoder.predict(enc.embed_window(neural[split - w:split])[None])

    # ---- the real-time loop ---------------------------------------------------------------
    buf = deque(neural[split - w + 1:split], maxlen=w)  # the w-1 samples just before the test set
    end = len(neural) if not args.steps else min(len(neural), split + args.steps)
    errors, times = [], []
    print(f"streaming {end - split} samples ({(end - split) * BIN_S:.0f} s), budget {BIN_S * 1000:.0f} ms per sample")
    start = time.perf_counter()
    try:
        for k, t in enumerate(range(split, end)):
            due = start + k * BIN_S
            while (now := time.perf_counter()) < due:  # wait for the next sample to "arrive"
                time.sleep(min(due - now, 0.002))
            t0 = time.perf_counter()
            buf.append(neural[t])                         # 1. new sample arrives
            e = enc.embed_window(np.stack(buf))           # 2. encoder on the last 10 samples
            pos, right = decoder.predict(e[None])         # 3. kNN decode
            pos, right = float(pos[0]), float(right[0])
            true_rgb = colors.rgb(label[t, 0], label[t, 1])
            cebra_rgb = colors.rgb(pos, right)
            if link:
                link.send(true_rgb, cebra_rgb)            # 4. update the LEDs
            times.append((time.perf_counter() - t0) * 1000)
            errors.append(abs(pos - label[t - delay, 0]))  # compare to the moment it describes
            if args.preview and k % 2 == 0:
                print(f"\r  true {ansi_block(true_rgb)}  cebra {ansi_block(cebra_rgb)}  "
                      f"pos {label[t, 0]:.2f} m vs {pos:.2f} m  {times[-1]:5.2f} ms ", end="", flush=True)
    except KeyboardInterrupt:
        print("\nstopped early")
    finally:
        if link:
            link.close()
    if args.preview:
        print()

    # ---- results ----------------------------------------------------------------------------
    times = np.array(times)
    summary = {
        "model": args.model, "control": args.control, "hardware": bool(link),
        "n_steps": len(times), "delay_ms": delay * BIN_S * 1000,
        "median_error_m": float(np.median(errors)),
        "latency_ms": {"mean": float(times.mean()), "p99": float(np.percentile(times, 99)),
                       "max": float(times.max())},
        "over_budget_steps": int(np.sum(times > BIN_S * 1000)),
        "ack_errors": link.ack_errors if link else None,
    }
    json.dump(summary, open(out / "run.json", "w"), indent=2)

    lat = summary["latency_ms"]
    print(f"\n===== {out.name} =====")
    print(f"median position error {summary['median_error_m']:.4f} m (decode is {summary['delay_ms']:.0f} ms behind)")
    print(f"time per sample: mean {lat['mean']:.2f} ms, p99 {lat['p99']:.2f} ms, max {lat['max']:.2f} ms "
          f"(budget {BIN_S * 1000:.0f} ms, over budget {summary['over_budget_steps']} times)")
    if link:
        print(f"ack errors: {link.ack_errors}")
    print(f"saved {out}/run.json")


if __name__ == "__main__":
    main()
