"""Evaluate shared diff_protocol requests with the fixed CUDA-Q/GPU Lanczos judge."""
from hsopt.campaign.protocol import main

if __name__ == "__main__":
    raise SystemExit(main(family="diff_protocol", worker=True))
