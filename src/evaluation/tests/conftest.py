"""
Pytest configuration and fixtures for evaluation tests.

This file configures pytest markers and shared fixtures for all test files.

Author: Your Name
Date: January 24, 2026
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", 
        "integration: mark test as integration test (may be slow or require full setup)"
    )
    config.addinivalue_line(
        "markers",
        "unit: mark test as unit test (fast, isolated)"
    )