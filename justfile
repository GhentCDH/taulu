alias t := test-all
alias ta := test-auto
alias b := build

# test this python project
test-all file='':
    uv run pytest {{file}}

test-visual file='':
    uv run pytest -m visual {{file}}

test-auto file='':
    uv run pytest -m "not visual" {{file}}

build:
    uv build
