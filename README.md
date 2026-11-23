# Real-Time CEBRA Decoding on Two RGB LEDs

Can CEBRA decode where a rat is from its brain activity in real time, as accurately as it does offline?

Uses the trained models from [CEBRA_Reproduce_Workflow](https://github.com/ZacharyPrager/CEBRA_Reproduce_Workflow). The held-out rat data is streamed one 25 ms sample at a time, and an ESP32 shows the result:

- **LED 1:** true position. **LED 2:** CEBRA's prediction.
- Color = position on the track, brightness = direction.

## Results

| | Real model | Control (shuffled) |
|---|---|---|
| Median position error | 0.077 m | 0.475 m |
| Mean time per sample (limit 25 ms) | 10.8 ms | 10.9 ms |
| Over 25 ms / packet errors | 0 / 0 | 0 / 0 |

Raw results are in `results/`.

## Setup

Tested with Python 3.14, CEBRA 0.6.1, CPU only (exact versions in `env/`).

```bash
bash setup_local.sh                      # Python environment + dataset
cd firmware && pio run -t upload         # flash the ESP32 (or Upload in VS Code PlatformIO)
```

Wiring: LED 1 on GPIO 25/26/27, LED 2 on GPIO 32/33/13 (R/G/B), each through a resistor. Common leg to 3V3 (common anode).

## Run

```bash
./run.sh --preview --steps 400                                                            # no hardware
./run.sh --port /dev/ttyUSB0 --ack                                                        # real model
./run.sh --model models/cebra_posdir3_shuffled_split.pt --control --port /dev/ttyUSB0 --ack  # control
```

On Linux, if the port says permission denied: `sudo chmod a+rw /dev/ttyUSB0`.
