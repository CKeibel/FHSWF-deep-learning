import argparse

from rag.utils import TrainingConfig


def parse_args() -> TrainingConfig:
    parser: argparse.Namespace = argparse.ArgumentParser(
        description="Causal Language Modeling Training Script"
    )
    # TODO: Add argument for config path
    pass


def main() -> None:
    pass


if __name__ == "__main__":
    main()
