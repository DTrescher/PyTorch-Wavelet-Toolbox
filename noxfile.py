"""This module implements our CI function calls."""
import nox

@nox.session(name="test")
def run_test(session):
    """Run pytest."""
    session.install(".")
    session.install("pytest")
    session.run("pytest")

# TODO: add not slow test

@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("flake8")
    session.install(
        "flake8-colors",
        "flake8-black",
        "flake8-docstrings",
        "flake8-bugbear",
        "flake8-broken-line",
        "pep8-naming",
        "pydocstyle",
        "darglint",
    )
    session.install("flake8-bandit==2.1.2", "bandit==1.7.2")
    session.run("flake8", "src", "tests", "noxfile.py")

@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install(".")
    session.install("mypy")
    session.run("mypy", "src", "tests")

@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("black", "src", "tests", "noxfile.py")
    session.run("isort", "src", "tests", "noxfile.py")

@nox.session(name="coverage")
def check_coverage(session):
    """Check test coverage and generate a html report."""
    session.install(".")
    session.install("pytest")
    session.install("coverage")
    try:
        session.run("coverage", "run", "-m", "pytest")
    finally:
        session.run("coverage", "html")

@nox.session(name="coverage-clean")
def clean_coverage(session):
    session.run("rm", "-r", "coverage_html_report",
                external=True)
