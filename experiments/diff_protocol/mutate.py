"""One shared diff_protocol mutation trajectory; results live outside the checkout."""
from hsopt.campaign.protocol import main

if __name__ == "__main__":
    raise SystemExit(main(family="diff_protocol"))
