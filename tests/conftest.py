"""
pytest configuration — shared fixtures and path setup.
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `import deepseek` works
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
