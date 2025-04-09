alias t := test
alias b := build

# test this python project
test:
    uv run pytest

build:
    uv build
