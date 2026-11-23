#!/bin/bash
# Runs the streamer with the project's Python (.venv), from any terminal, no activate needed:
#   ./run.sh --port /dev/ttyUSB0 --ack
#   ./run.sh --model models/cebra_posdir3_shuffled_split.pt --control --port /dev/ttyUSB0 --ack
cd "$(dirname "$(readlink -f "$0")")"
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found. Run:  bash setup_local.sh" >&2
  exit 1
fi
exec .venv/bin/python -W ignore stream/stream_decode.py "$@"
