alias t := test-all
alias ta := test-auto
alias b := build

# test this python project
test-all file='':
    uv run pytest {{file}}

test-debug file='':
    uv run pytest {{file}} --log-cli-level=DEBUG -v -s

test-visual file='':
    uv run pytest -m visual {{file}}

test-auto file='':
    uv run pytest -m "not visual" {{file}}

build:
    uv build

profile-astar:
    timeout 10 uv run py-spy record --native -r 50 -o profile.svg -- python tests/bench_rust.py
