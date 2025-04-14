#!/bin/bash

cp ../data/header_00.png header.png
cp ../data/table_00.png table.png


if ! command -v uv > /dev/null; then
    printf "\x1b[1mYou need to install uv in order to run this example\x1b[0m\n"
    printf "or you can install the taulu dependcy using pip:\n"
    printf "\tpip install git+https://github.com/ghentcdh/taulu.git\n"
    printf "\tpython example.py\n"

    exit 1
fi

if [ ! -f pyproject.toml ]; then
    echo "Initializing python uv project"
    uv init --no-workspace --bare
    uv add ..
fi

uv run example.py 
