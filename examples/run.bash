#!/bin/bash

cp ../data/header_00.png header.png
cp ../data/table_00.png table.png

if ! command -v uv > /dev/null; then
    echo "you need to install uv in order to use this script"
fi

if [ ! -f pyproject.toml ]; then
    echo "Initializing python uv project"
    uv init --no-workspace
    uv add ..
fi

uv run example.py 
