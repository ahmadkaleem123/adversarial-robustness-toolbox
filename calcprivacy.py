import pickle

dataset = "svhn"
NB_STOLEN = 1000

with open(f"selectedidxs/{dataset}ind{NB_STOLEN}", "rb") as fp:
    selected_idxs = pickle.load(fp)

print("selected idxs", selected_idxs)
print("len", len(selected_idxs))