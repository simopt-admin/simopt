"""Command-line interface for SimOpt."""

import click


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Run SimOpt command-line tools."""


@main.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Interface to bind the web server to.",
)
@click.option(
    "--port",
    default=8000,
    show_default=True,
    type=click.IntRange(min=1, max=65535),
    help="Port to bind the web server to.",
)
@click.option(
    "--reload/--no-reload",
    "reload_enabled",
    default=True,
    show_default=True,
    help="Restart the server when source files change.",
)
@click.option(
    "--log-level",
    default="info",
    show_default=True,
    type=click.Choice(
        ["critical", "error", "warning", "info", "debug", "trace"],
        case_sensitive=False,
    ),
    help="Uvicorn log verbosity.",
)
def web(host: str, port: int, reload_enabled: bool, log_level: str) -> None:
    """Start the SimOpt FastAPI web interface."""
    import uvicorn

    click.echo("Starting SimOpt Web Interface...")
    click.echo(f"SimOpt is running at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop.")
    uvicorn.run(
        "simopt.web.server:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
