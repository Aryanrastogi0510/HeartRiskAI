"""Smoke tests for Streamlit application (app.py) using AppTest."""

import pytest
from streamlit.testing.v1 import AppTest


def test_app_renders_risk_assessment_page():
    """Verifies that app.py loads and renders the primary Risk Assessment page without errors."""
    at = AppTest.from_file("app.py", default_timeout=20)
    at.run()
    assert not at.exception


def test_app_page_navigation():
    """Verifies that navigation to each of the four pages completes without exceptions."""
    pages = [
        "🩺 1. Risk Assessment",
        "📊 2. Model Comparison",
        "🔍 3. Risk Factor Explanation",
        "📖 4. About the Project",
    ]
    for page_name in pages:
        at = AppTest.from_file("app.py", default_timeout=20)
        at.run()
        at.sidebar.radio[0].set_value(page_name)
        at.run()
        assert not at.exception
