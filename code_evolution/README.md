### Usage

To profile with `legion_prof` you need to install with the `config_profiler.sh` script. Then, run 
```
CUDA_VISIBLE_DEVICES=0 uv run  legate --profile initial_program.py
```
And change `CUDA_VISIBLE_DEVICES` to the GPU you want to profile on. This will generate a `legate_0.prof` file that you can analyze with `uv run legion_prof view legate_*.prof `. However, to extract a simple number for our cost function, do the following:

```
CUDA_VISIBLE_DEVICES=0 nsys profile -o prof --export=sqlite --trace=cuda,nvtx,osrt --sample=none uv run legate --profile initial_program.py
``` 
This will generate a `prof.sqlite` file that you can analyze with the command:
```
uv run gpu_cost.py 
```