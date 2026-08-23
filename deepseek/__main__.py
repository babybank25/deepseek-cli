"""Entry point: python -m deepseek."""
import argparse
import asyncio
import json

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
        help="Open browser to discover API/auth state and save config",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Re-run browser discovery when protocol/auth recovery cannot self-heal",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Side-effect-free auth/PoW protocol capability check",
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
        help="Run browser headless during --discover/--repair",
    )
    parser.add_argument("--serve", action="store_true", help="Start compatibility API server")
    parser.add_argument("--host", default="127.0.0.1", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override the persistent/generated local API key",
    )
    parser.add_argument(
        "--gateway",
        action="store_true",
        help="Route through all saved accounts with affinity-aware scheduling",
    )
    parser.add_argument(
        "--admin-token",
        default=None,
        help="X-Admin-Token for /admin/* endpoints; admin routes are disabled if unset",
    )
    parser.add_argument(
        "--auto-compact",
        type=int,
        default=None,
        metavar="N",
        help="Auto-compact threshold; 0 disables, unset keeps package default",
    )
    parser.add_argument("--chat", action="store_true", help="Launch terminal chat")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    return parser


async def _probe(args, console) -> None:
    from .protocol import probe_protocol

    if args.gateway:
        from .gateway import AccountPool

        pool = AccountPool()
        pool.load_all()
        if pool.is_empty():
            console.print("[red]No gateway accounts configured.[/]")
            return
        results = []
        for account in pool.accounts:
            capabilities = await probe_protocol(account.config)
            results.append({"account": account.name, **capabilities.__dict__})
        await pool.close_all()
        console.print_json(json.dumps({"results": results}, ensure_ascii=False))
        return

    config = SessionManager.load_config()
    if not config:
        console.print("[red]No saved config. Run --discover first.[/]")
        return
    capabilities = await probe_protocol(config)
    console.print_json(json.dumps(capabilities.__dict__, ensure_ascii=False))


async def main() -> None:
    from rich.console import Console

    console = Console()
    args = build_parser().parse_args()

    if args.version:
        console.print(f"DeepSeek Web CLI v{VERSION}")
        return
    if args.probe:
        await _probe(args, console)
        return
    if args.discover or args.repair:
        from .discover import discover_mode

        await discover_mode(args.target, headless=args.headless)
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
    if SessionManager.config_exists():
        from .cli import chat_mode

        await chat_mode()
        return

    console.print("[yellow]No saved config. Running first-time discovery…[/]")
    from .discover import discover_mode
    from .cli import chat_mode

    await discover_mode(args.target, headless=args.headless)
    if SessionManager.config_exists():
        await chat_mode()


if __name__ == "__main__":
    asyncio.run(main())
