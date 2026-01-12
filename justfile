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
    timeout 40 uv run py-spy record --native -r 200 -o profile.svg --subprocesses -f speedscope -- python tests/bench_rust.py

document:
    uv run pdoc --logo './logo.svg' --favicon './favicon.svg' -o ./docs --docformat google taulu

setup-rerun:
    uv run maturin develop --features debug-tools --no-default-features --release
