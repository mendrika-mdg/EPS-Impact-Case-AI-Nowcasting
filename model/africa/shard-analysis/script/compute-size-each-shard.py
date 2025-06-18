import os, json, torch, tqdm, sys

PARTITION = sys.argv[1]
LEAD_TIME = sys.argv[2]

SHARDS_DIR = f"/work/scratch-nopw2/mrakotomanga/eps/pancast-shard-processed/t{LEAD_TIME}/{PARTITION}"
OUTPUTS_DIR = f"/home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/model/africa/shard-analysis/t{LEAD_TIME}/{PARTITION}"

sizes = {}
for fn in tqdm.tqdm(sorted(os.listdir(SHARDS_DIR))):
    if not fn.endswith("_proc.pt"):                       # skip stray files
        continue
    path = os.path.join(SHARDS_DIR, fn)
    # Only load the metadata, not the whole dict
    n = torch.load(path, map_location="cpu")["inputs"].shape[0]
    sizes[fn] = n

with open(os.path.join(OUTPUTS_DIR, f"{PARTITION}-t{LEAD_TIME}-shards-proc-sizes.json"), "w") as fp:
    json.dump(sizes, fp)
print("Wrote sizes.json with", len(sizes), "entries")
