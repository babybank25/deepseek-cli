from unittest.mock import patch

import deepseek.__main__ as cli_main


def test_run_executes_async_main_once():
    with patch("deepseek.__main__.asyncio.run") as run_async:
        cli_main.run()
    run_async.assert_called_once()
    coroutine = run_async.call_args.args[0]
    assert coroutine.cr_code.co_name == "main"
    coroutine.close()
