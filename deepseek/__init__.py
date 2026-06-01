"""
deepseek — DeepSeek Web CLI package.

Public API surface:
  from deepseek.client import APIClient
  from deepseek.models import APIConfig
  from deepseek.session import SessionManager
  from deepseek.exceptions import AuthExpiredError
"""
from .constants import VERSION

__version__ = VERSION
__all__ = ["VERSION"]
