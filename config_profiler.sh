cd ..
git clone https://github.com/StanfordLegion/legion.git
cd legion
git fetch --tags
git checkout legion-25.06.0
cargo install --locked --all-features --path tools/legion_prof_rs
cd ..