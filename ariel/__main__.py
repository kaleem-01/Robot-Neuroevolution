"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Ariel."""


if __name__ == "__main__":
    main(prog_name="ariel")  # pragma: no cover
