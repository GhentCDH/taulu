
#!/bin/bash

#pip install uv 

cp ../data/header_00.png header.png
cp ../data/table.png table.png
uv init
uv add ..
uv run example.py 
