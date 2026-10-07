"""Primary CLI entry point for reward-guided TX deployment sampling."""

from energy_model.infer import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()

