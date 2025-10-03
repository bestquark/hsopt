# pip install moviepy imageio imageio-ffmpeg
import mdtraj as md
import nglview as nv
from nglview.contrib.movie import MovieMaker
from IPython.display import display

# --- load + view ---
t = md.load_dcd("../sims/traj.dcd", top="../sims/final.pdb")
# (optional) subsample to speed up: t = t[::2]
v = nv.show_mdtraj(t)
v.clear_representations()
v.add_representation("cartoon", selection="protein")
v.add_representation("ball+stick", selection="not protein and not water")
v.center()
v._remote_call("setSize", target="Widget", args=["1280px", "720px"])  # 720p
display(v)  # MUST display so the frontend can render frames

print(f"Trajectory has {v.max_frame + 1} frames")

# --- make a quick movie (keeps it fast + reliable) ---
mm = MovieMaker(
    v,
    output="traj.mov",  # or "traj.gif"
    fps=24,
    start=0,
    stop=min(600, v.max_frame + 1),  # ~600 frames max for speed
    step=1,  # every other frame
)

mm.make()  # shows a progress bar; keeps running until "Done"
