"""One shared abs_protocol mutation trajectory; results live outside the checkout."""
from hsopt.campaign.protocol import main

if __name__ == "__main__":
    raise SystemExit(main(family="abs_protocol"))
