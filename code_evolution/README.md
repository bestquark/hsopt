### First Step
First, configure the number of GPUs to use with:
```bash
module load cuda/12.2 # or your desired CUDA version
source config_gpus.env.sh 
```
### Legion Profiling

To profile with `legion_prof` you need to install with the `config_profiler.sh` script. Then, run 
```
uv run legate --profile initial_program.py
```
This will generate a `legate_0.prof` file that you can analyze with `uv run legion_prof view legate_*.prof `. 

### Nsys Profiling

However, to extract a simple number for our cost function, do the following:

```
nsys profile -o prof --export=sqlite --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --force-overwrite true uv run legate --gpus 1 --profile initial_program.py
``` 
This will generate a `prof.sqlite` file that you can analyze with the command:
```
uv run gpu_cost.py 
```