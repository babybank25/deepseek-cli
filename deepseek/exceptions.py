"""
Custom exceptions for the DeepSeek CLI.

Public exceptions (AuthExpiredError) are raised to callers.
Private exceptions (_PowExpiredError, _SessionNotFoundError) are internal
signals used within APIClient's retry loops and should not escape the client.
"""


class AuthExpiredError(Exception):
    """Auth token or session has expired — user must run /reauth."""


class _PowExpiredError(Exception):
    """Internal: PoW token rejected by server — caller should re-solve and retry."""


class _SessionNotFoundError(Exception):
    """Internal: session_id is no longer valid — caller should recreate session."""
