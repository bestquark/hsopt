To run with GPUs, first configure with 
```bash 
source config_gpus.env.sh
```
Then run the desired script, e.g.
```bash 
uv run legate --gpus 1  --fbmem 256 testing_scripts/scripts/xyz.py
```