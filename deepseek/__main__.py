"""
Entry point: python -m deepseek
"""
import asyncio
import argparse

from .constants import DEFAULT_TARGET_URL, VERSION
from .session import SessionManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepseek",
        description="DeepSeek Web CLI — reverse-engineered terminal chat & API server",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Open browser to sniff API & save config",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET_URL,
        help=f"Target URL (default: {DEFAULT_TARGET_URL})",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode during --discover",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start an OpenAI-compatible HTTP API server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="API server host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional bearer token clients must send (default: no auth)",
    )
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="Route requests through multi-account pool (load all saved accounts)",
    )
    parser.add_argument(
        "--admin-token",
        type=str,
        default=None,
        help="Token for X-Admin-Token header on /admin/* endpoints (disabled by default)",
    )
    parser.add_argument(
        "--auto-compact",
        type=int,
        default=None,
        metavar="N",
        help="Auto-compact threshold (turns) for every client. 0 disables. "
             "If unset, uses constants.AUTO_COMPACT_THRESHOLD.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Launch TUI chat directly",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    return parser


async def main() -> None:
    from rich.console import Console
    console = Console()

    args = build_parser().parse_args()

    if args.version:
        console.print(f"DeepSeek Web CLI v{VERSION}")
        return

    if args.chat:
        from .cli import chat_mode
        await chat_mode(auto_compact_threshold=args.auto_compact)
        return

    if args.serve:
        from .server import serve_mode
        await serve_mode(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            use_gateway=args.gateway,
            admin_token=args.admin_token,
            auto_compact_threshold=args.auto_compact,
        )
        return

    if args.discover:
        from .discover import discover_mode
        await discover_mode(args.target, headless=args.headless)
    elif SessionManager.config_exists():
        from .cli import chat_mode
        await chat_mode()
    else:
        console.print("[yellow]No saved config. Running discovery mode first...[/]")
        from .discover import discover_mode
        from .cli import chat_mode
        await discover_mode(args.target, headless=args.headless)
        await chat_mode()


if __name__ == "__main__":
    asyncio.run(main())
