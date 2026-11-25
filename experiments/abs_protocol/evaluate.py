"""Evaluate shared abs_protocol requests with the fixed CUDA-Q/GPU Lanczos judge."""
from hsopt.campaign.protocol import main

if __name__ == "__main__":
    raise SystemExit(main(family="abs_protocol", worker=True))
