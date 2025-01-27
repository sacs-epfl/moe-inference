import json
import os
import math

def list_dirs(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def list_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)



for base_d in ["mgblcks", "sliced"]:
    for d in list_dirs(f"./{base_d}"):
        rs = None
        if d.split("_")[-1] == "rs":
            rs = float(d.split("_")[-2])
        else:
            rs = 0.6


        data = read_json(f"./{base_d}/{d}/data.json")
        data["router_skew"] = rs
        data["num_experts_skew"] = math.ceil(data["num_experts"] * 0.1)
        data["random_router_skew"] = False
        data["enable_router_skew"] = True

        write_json(f"./{base_d}/{d}/data.json", data)
    
    