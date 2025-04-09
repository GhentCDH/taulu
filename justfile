alias t := test-all
alias b := build

# test this python project
test-all:
    uv run pytest

test-visual:
    uv run pytest -m visual

test-auto:
    uv run pytest -m "not visual"

build:
    uv build
