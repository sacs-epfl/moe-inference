import sys
import pandas as pd
import json
import os
import stat
from datetime import datetime
import argparse
import ast
import numpy as np
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from theme import update_fig_to_theme, colours

import plotly.express as px
from typing import List, Dict

OUTPUT_DIR = f"../plots/{datetime.today().strftime('%Y-%m-%d_%H-%M')}"

def list_folders(path: str):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def list_files(path: str):
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]

def create_dir_if_needed():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        os.chmod(OUTPUT_DIR, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

if len(sys.argv) < 2:
    print(f"Usage: python3 {sys.argv[0]} num_gpus paths")

def read_data(_dir: str):
    with open(f"{_dir}/data.json", "r") as f:
        return json.load(f)

def plot_e2e(dirs: [str]):
    frames = []
    for _dir in dirs:
        df = pd.read_csv(f"{_dir}/0/e2e.csv")
        data = read_data(_dir)
        print(f"Dir {_dir} - Data Name: {data['name']}")
        df["name"] = data["label"]

        frames.append(df)

    df = pd.concat(frames)
    df = df[df["Iteration Number"] > 3]
    def num_exp(x):
        x = x.split(" ")
        for i in x:
            if i.isdigit():
                return int(i)

    df["sort_key"] = df["name"].apply(lambda x: num_exp(x))
    df = df.sort_values(["sort_key", "name", "Iteration Number"])

    fig = px.line(df, x="Iteration Number", y="Latency (s)", color="name")
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/e2e.pdf", format="pdf")


def plot_average_bytes_sent_received(dirs: [str]):
    if len(dirs) == 1:
        frames = []
        data = read_data(dirs[0])
        i = 0
        while (os.path.exists(f"{dirs[0]}/{i}/")):
            j = 1
            frames_names = []
            while (os.path.exists(f"{dirs[0]}/{i}/moe_l{j}.csv")):
                frames_names.append(f"{dirs[0]}/{i}/moe_l{j}.csv")
                j += 2
            
            main_df = pd.DataFrame()
            for frame_name in frames_names:
                df = pd.read_csv(frame_name)
                if main_df.empty:
                    main_df = df.copy()
                else:
                    main_df["total number of bytes sent"] += df["total number of bytes sent"] 
                    main_df["total number of bytes recv"] += df["total number of bytes recv"]

            main_df["total number of bytes sent"] = main_df["total number of bytes sent"] / len(frames_names)
            main_df["total number of bytes recv"] = main_df["total number of bytes recv"] / len(frames_names)
            main_df["name"] = "GPU {}".format(i)

            frames.append(main_df)
            i += 1
        
        df = pd.concat(frames)
        print(type(df))
        df = df.sort_values(["name", "iteration"])
        df = df[df["iteration"] > 3]
        fig = px.line(df, x="iteration", y="total number of bytes sent", color="name")
        update_fig_to_theme(fig, title="Tot bytes sent for {}".format(data["label"]),
                             xaxis="Iteration Number", yaxis="Number of Bytes")

        create_dir_if_needed()
        fig.write_image(f"{OUTPUT_DIR}/tot_bytes_sent_comms_{data['label']}.png")

        fig = px.line(df, x="iteration", y="total number of bytes recv", color="name")
        update_fig_to_theme(fig, title="Tot bytes recv for {}".format(data["label"]),
                             xaxis="Iteration Number", yaxis="Number of Bytes")

        create_dir_if_needed()
        fig.write_image(f"{OUTPUT_DIR}/tot_bytes_recv_comms_{data['label']}.png")
        return

                
                
    frames = []
    for _dir in dirs:
        main_df = pd.DataFrame()
        data = read_data(_dir)
        i = 1
        frames_names = []
        while (os.path.exists(f"{_dir}/0/moe_l{i}.csv")):
            frames_names.append(f"{_dir}/0/moe_l{i}.csv")
            i += 2
        for frame_name in frames_names:
            df = pd.read_csv(frame_name)
            if main_df.empty:
                main_df = df.copy()
            else:
                main_df["total number of bytes sent"] += df["total number of bytes sent"] 
                main_df["total number of bytes recv"] += df["total number of bytes recv"]
        main_df["total number of bytes sent"] = main_df["total number of bytes sent"] / len(frames_names)
        main_df["total number of bytes recv"] = main_df["total number of bytes recv"] / len(frames_names)
        main_df["name"] = data["label"]

        frames.append(main_df)

    df = pd.concat(frames)
    df = df.sort_values(["name", "iteration"])
    df = df[df["iteration"] > 3]

    fig = px.line(df, x="iteration", y="total number of bytes sent", color="name")
    update_fig_to_theme(fig, title="Total number of bytes sent (averaged across layers)",
        xaxis="Iteration Number", yaxis="Number of Bytes")    

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/tot_bytes_sent_comms.png")

    fig = px.line(df, x="iteration", y="total number of bytes recv", color="name")
    update_fig_to_theme(fig, title="Total number of bytes recv (averaged across layers)",
        xaxis="Iteration Number", yaxis="Number of Bytes")    

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/tot_bytes_recv_comms.png")
    

def plot_average_speedup_evolution(dirs, evolution_type, evolution_label):
    def list_folders(path: str):
        return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

    model_mapping = {"simple": "DeepSpeed", "manual_sliced": "Sliced", "sliced_fused_kernel": "MegaBlocks"}
    model_type_mapping = {"encoder": "Encoder", "encoder-decoder": "Encoder-Decoder"}
    frames = []
    for _dir in dirs:
        if _dir == "":
            continue
        data = read_data(_dir)
        print(f"Model: {model_mapping[data['expert_manager']]} - Model Type: {model_type_mapping[data['model_type']]} - {evolution_label}: {data[evolution_type]} - Name: {data['name']}")
        frames_exp = []
        for gpu in list_folders(_dir):
            df = pd.read_csv(os.path.join(_dir, gpu, "e2e.csv"))
            df = df[df["Iteration Number"] > 3]
            df = df.rename(columns={"Latency (s)": f"GPU {gpu}"})

            frames_exp.append(df)

        if len(frames_exp) < data["world_size"]:
            raise ValueError(f"Missing data for {_dir}")

        exp_df = frames_exp[0]
        for df in frames_exp[1:]:
            exp_df = exp_df.merge(df, on="Iteration Number", how="outer")

        exp_df["model"] = model_mapping[data["expert_manager"]]
        exp_df["model_type"] = model_type_mapping[data["model_type"]]
        exp_df["val"] = exp_df.loc[:, ["GPU {}".format(i) for i in range(data["world_size"])]].mean(axis=1)
        try:
            exp_df[evolution_label] = int(data[evolution_type])
        except:
            print("Could not convert to int!!!!!!!!!!")
            exp_df[evolution_label] = data[evolution_type]


        frames.append(exp_df[["model", "model_type", "val", "Iteration Number", evolution_label]])
        

    if len(frames) < 1:
        print("No data to work on, finishing plot_average_speedup")
        return
    
    comb_df = pd.concat(frames, axis=0, ignore_index=True)

    comb_df["speedup"] = 0.0
    for mt in comb_df["model_type"].unique():
        for it in comb_df["Iteration Number"].unique():
            for ev in comb_df[evolution_label].unique():
                comparison = comb_df[(comb_df["model"] == "DeepSpeed") & (comb_df["model_type"] == mt) & (comb_df["Iteration Number"] == it) & (comb_df[evolution_label] == ev)]["val"].values[0]

                comb_df.loc[(comb_df["model_type"] == mt) & (comb_df["Iteration Number"] == it) & (comb_df[evolution_label] == ev), "speedup"] = comparison / comb_df[(comb_df["model_type"] == mt) & (comb_df["Iteration Number"] == it) & (comb_df[evolution_label] == ev)]["val"]
    

    avg_df = comb_df.groupby(["model", "model_type", evolution_label])["speedup"].mean().reset_index()
    avg_df["speedup"] = avg_df["speedup"].round(2)

    fig = px.line(
        avg_df,
        x=evolution_label,
        y="speedup", 
        color="model", 
        color_discrete_map={"DeepSpeed": "gray", "Sliced": "red", "MegaBlocks": "blue"},
        line_dash="model_type", 
        markers=True, 
        text="speedup", 
        labels={"model": "Model", "model_type": "Model Type"}
    )
    fig.update_traces(textposition='top center')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=True)
    update_fig_to_theme(fig, xaxis=evolution_label, yaxis="Average Speedup")
    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/average_speedup_evolution.pdf", format="pdf")



def model_mapping(label):
    if "DeepSpeed" in label:
        return "DeepSpeed"
    elif "Sliced" in label or "manual_sliced" in label:
        return "Sliced"
    elif "MegaBlocks" in label or "sliced_fused_kernel" in label:
        return "MegaBlocks"
    
    else:
        raise ValueError(f"Could not map {label}")

def model_type_mapping(model_type):
    if model_type == "encoder":
        return "Encoder"
    elif model_type == "encoder-decoder":
        return "Encoder-Decoder"
    else:
        raise ValueError(f"Could not map {model_type}")
    

def plot_average_speedup_evolution_2(dirs, evolution_type, evolution_label):
    def process_dir(_dir):
        if _dir == "":
            return None
        data = read_data(_dir)
        print(f"Model: {model_mapping(data['label'])} - Model Type: {model_type_mapping(data['model_type'])} - {evolution_label}: {data[evolution_type]} - Name: {data['name']if 'name' in data else data['experiment_name']}")
        frames_same_layer = defaultdict(list)
        for gpu in list_folders(_dir):
            for layer in list_files(os.path.join(_dir, gpu)):
                if layer == "e2e.csv":
                    continue
                df = pd.read_csv(os.path.join(_dir, gpu, layer))
                df = df[df["iteration"] > 3]
                df.rename(columns={"latency (ms)": f"GPU {gpu}"}, inplace=True)
                df = df[["iteration", f"GPU {gpu}"]]
                frames_same_layer[layer.split(".")[0]].append(df)
        
        f_df = []
        for layer, dfs in frames_same_layer.items():
            l_df = dfs[0]
            for df in dfs[1:]:
                l_df = l_df.merge(df, on="iteration", how="outer")
            
            l_df["val"] = l_df.iloc[:, l_df.columns != "iteration"].mean(axis=1)
            l_df["layer"] = layer
            f_df.append(l_df)
        
        f_df = pd.concat([d[["iteration", "val", "layer"]] for d in f_df], axis=0, ignore_index=True)
        f_df["model"] = model_mapping(data["label"])
        f_df["model_type"] = model_type_mapping(data["model_type"])
        try:
            f_df[evolution_label] = float(data[evolution_type])
        except:
            print("Could not convert to int!!!!!!!!!!")
            f_df[evolution_label] = data[evolution_type]
        
        return f_df

    with ThreadPoolExecutor(max_workers=len(dirs)) as executor:
        frames = list(executor.map(process_dir, dirs))
    
    frames = [f for f in frames if f is not None]
    
    if len(frames) < 1:
        print("No data to work on, finishing plot_average_speedup")
        return
    frames = pd.concat(frames, axis=0, ignore_index=True)

    speedups = []

    frames = frames.groupby(["model_type", "layer", evolution_label])

    for (model_type, layer, evolution), group in frames:

        comparison = group[(group["model"] == "DeepSpeed")]["val"]
        assert len(comparison) == 96, f"Length of comparison: {len(comparison)}"
        sliced = group[(group["model"] == "Sliced")]["val"]
        assert len(sliced) == 96, f"Length of sliced: {len(sliced)}"
        mega = group[(group["model"] == "MegaBlocks")]["val"]
        assert len(mega) == 96, f"Length of mega: {len(mega)}"

        assert len(group) == 96 * 3

        comparison = comparison.mean()
        sliced = sliced.mean()
        mega = mega.mean()

        sliced_speedup = comparison / sliced
        mega_speedup = comparison / mega

        new_speedups = pd.DataFrame([
        {
            "model": "DeepSpeed",
            "model_type": model_type,
            evolution_label: evolution,
            "speedup": 1.0,
            "layer": layer
        },
        {
            "model": "MegaBlocks",
            "model_type": model_type,
            evolution_label: evolution,
            "speedup": mega_speedup,
            "layer": layer
        },
        {
            "model": "Sliced",
            "model_type": model_type,
            evolution_label: evolution,
            "speedup": sliced_speedup,
            "layer": layer
        }])
        speedups.append(new_speedups)
    
    speedups = pd.concat(speedups, axis=0, ignore_index=True)

    avg_speedups = speedups.groupby(["model", "model_type", evolution_label])["speedup"].mean().reset_index()
    avg_speedups["speedup"] = avg_speedups["speedup"].round(2)

    fig = px.line(
        avg_speedups,
        x=evolution_label,
        y="speedup",
        color="model",
        color_discrete_map={"DeepSpeed": "gray", "Sliced": "red", "MegaBlocks": "blue"},
        line_dash="model_type",
        markers=True,
        text="speedup",
        labels={"model": "Model", "model_type": "Model Type"}
    )

    fig.update_traces(textposition='top center')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide", showlegend=True)
    update_fig_to_theme(fig, xaxis=evolution_label, yaxis="Average Speedup")
    

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/average_speedup_evolution.pdf", format="pdf")
    avg_speedups[evolution_label] = avg_speedups[evolution_label].astype(int)
    avg_speedups.to_csv(f"{OUTPUT_DIR}/average_speedup_evolution.csv", index=False)


def save_merge_csv_evolution(dirs, evolution_type, evolution_label):
    def process_dir(_dir):
        if _dir == "":
            return None
        data = read_data(_dir)
        print(f"Model: {model_mapping(data['label'])} - Model Type: {model_type_mapping(data['model_type'])} - {evolution_label}: {data[evolution_type]} - Name: {data['name']if 'name' in data else data['experiment_name']}")
        total_dir_df = []
        for gpu in list_folders(_dir):
            for layer in list_files(os.path.join(_dir, gpu)):
                if layer == "e2e.csv":
                    continue
                df = pd.read_csv(os.path.join(_dir, gpu, layer))
                df["gpu"] = gpu
                df["layer"] = layer.split(".")[0]
                df = df[["iteration", "latency (ms)", "gpu", "layer"]]
                total_dir_df.append(df)
        
        total_dir_df = pd.concat(total_dir_df, axis=0, ignore_index=True)
        total_dir_df["model_type"] = model_type_mapping(data["model_type"])
        total_dir_df["model"] = model_mapping(data["label"])
        total_dir_df[evolution_label] = data[evolution_type]

        return total_dir_df

    with ThreadPoolExecutor(max_workers=len(dirs)) as executor:
        final_df = pd.concat(list(executor.map(process_dir, dirs)), axis=0, ignore_index=True)
    
    print(final_df.columns)
    print(final_df.sample(frac=1, random_state=42).head(20))

    expected_size = final_df["iteration"].nunique() * final_df["layer"].nunique() * final_df["model"].nunique() * final_df[evolution_label].nunique() * final_df["gpu"].nunique() + final_df["iteration"].nunique() * final_df["layer"].nunique() / 2 * final_df["model"].nunique() * final_df[evolution_label].nunique() * final_df["gpu"].nunique()

    print(f"Num layers: {final_df['layer'].nunique()}")
    print(f"Num iterations: {final_df['iteration'].nunique()}")
    print(f"Num models: {final_df['model'].nunique()}")
    print(f"Num model types: {final_df['model_type'].nunique()}")
    print(f"Num {evolution_label}: {final_df[evolution_label].nunique()}")
    print(f"Num GPUs: {final_df['gpu'].nunique()}")

    assert len(final_df) == expected_size, "Expected size: {} - Actual size: {}".format(expected_size, len(final_df))

    create_dir_if_needed()
    final_df.sort_values([evolution_label, "model", "model_type", "gpu", "layer", "iteration"]).to_csv(f"{OUTPUT_DIR}/merged.csv", index=False)

def save_merge_csv_e2e(dirs):
    def process_dir(_dir):
        if _dir == "":
            return None
        data = read_data(_dir)
        print(f"Model: {model_mapping(data['label'])} - Model Type: {model_type_mapping(data['model_type'])} - Name: {data['name']if 'name' in data else data['experiment_name']}")
        total_dir_df = []
        for gpu in list_folders(_dir):
            for layer in list_files(os.path.join(_dir, gpu)):
                if layer == "e2e.csv":
                    continue
                df = pd.read_csv(os.path.join(_dir, gpu, layer))
                df["gpu"] = gpu
                df["layer"] = layer.split(".")[0]
                df = df[["iteration", "latency (ms)", "gpu", "layer"]]
                total_dir_df.append(df)
        
        total_dir_df = pd.concat(total_dir_df, axis=0, ignore_index=True)
        total_dir_df["model_type"] = model_type_mapping(data["model_type"])
        total_dir_df["model"] = model_mapping(data["label"])

        return total_dir_df

    with ThreadPoolExecutor(max_workers=len(dirs)) as executor:
        final_df = pd.concat(list(executor.map(process_dir, dirs)), axis=0, ignore_index=True)

    print(final_df.columns)
    print(final_df.sample(frac=1, random_state=42).head(20))

    len_dirs = len(list(d for d in dirs if d != ''))
    expected_size = final_df["iteration"].nunique() * final_df["layer"].nunique() * final_df["gpu"].nunique() * len_dirs 

    print(f"Num layers: {final_df['layer'].nunique()}")
    print(f"Num iterations: {final_df['iteration'].nunique()}")
    print(f"Num models: {final_df['model'].nunique()}")
    print(f"Num model types: {final_df['model_type'].nunique()}")
    print(f"Num GPUs: {final_df['gpu'].nunique()}")
    print(f"Num dirs: {len_dirs}")

    assert len(final_df) == expected_size, "Expected size: {} - Actual size: {}".format(expected_size, len(final_df))

    create_dir_if_needed()
    final_df.sort_values(["model", "model_type", "gpu", "layer", "iteration"]).to_csv(f"{OUTPUT_DIR}/merged.csv", index=False)

def save_merge_csv_time_section(dirs):
    columns_of_interest = ["Before Metadata and 1st Data", "Metadata and 1st Data", "Before 2nd Data", "2nd Data", "After 2nd Data"]
    def process_dir(_dir):
        if _dir == "":
            return None
        data = read_data(_dir)
        print(f"Model: {model_mapping(data['label'])} - Model Type: {model_type_mapping(data['model_type'])} - Name: {data['name']if 'name' in data else data['experiment_name']}")
        total_dir_df = []
        for gpu in list_folders(_dir):
            for layer in list_files(os.path.join(_dir, gpu)):
                if layer == "e2e.csv":
                    continue
                df = pd.read_csv(os.path.join(_dir, gpu, layer))
                df["gpu"] = gpu
                df["layer"] = layer.split(".")[0]
                df = df[["iteration", "gpu", "layer"] + columns_of_interest]
                total_dir_df.append(df)
        
        total_dir_df = pd.concat(total_dir_df, axis=0, ignore_index=True)
        total_dir_df["model_type"] = model_type_mapping(data["model_type"])
        total_dir_df["model"] = model_mapping(data["label"])

        return total_dir_df

    with ThreadPoolExecutor(max_workers=len(dirs)) as executor:
        final_df = pd.concat(list(executor.map(process_dir, dirs)), axis=0, ignore_index=True)

    print(final_df.columns)
    print(final_df.sample(frac=1, random_state=42).head(20))

    len_dirs = len(list(d for d in dirs if d != ''))
    expected_size = final_df["iteration"].nunique() * final_df["layer"].nunique() * final_df["gpu"].nunique() * len_dirs 

    print(f"Num layers: {final_df['layer'].nunique()}")
    print(f"Num iterations: {final_df['iteration'].nunique()}")
    print(f"Num models: {final_df['model'].nunique()}")
    print(f"Num model types: {final_df['model_type'].nunique()}")
    print(f"Num GPUs: {final_df['gpu'].nunique()}")
    print(f"Num dirs: {len_dirs}")

    assert len(final_df) == expected_size, "Expected size: {} - Actual size: {}".format(expected_size, len(final_df))

    create_dir_if_needed()
    final_df.sort_values(["model", "model_type", "gpu", "layer", "iteration"]).to_csv(f"{OUTPUT_DIR}/merged.csv", index=False)
    

def plot_speedup_across_dataset(dirs: [str]):
    exp_frames = []
    for _dir in dirs:
        if _dir == "":
            continue
        data = read_data(_dir)
        frames = []
        for gpu in list_folders(_dir):
            df = pd.read_csv(os.path.join(_dir, gpu, "e2e.csv"))
            df = df[df["Iteration Number"] > 3]
            df.rename(columns={"Latency (s)": f"GPU {gpu}"}, inplace=True)
            frames.append(df)
        
        df_exp = frames[0]
        for df in frames[1:]:
            df_exp = df_exp.merge(df, on="Iteration Number", how="outer")
        

        df_exp["val"]  = df_exp.iloc[:, df_exp.columns != "Iteration Number"].mean(axis=1)
        df_exp["std_between_gpus"] = df_exp.iloc[:, df_exp.columns != "Iteration Number"].std(axis=1)

        df_exp["dataset"] = data["dataset"]
        df_exp["model_type"] = data["model_type"]
        df_exp["model"] = data["expert_manager"]

        exp_frames.append(df_exp[["dataset", "model_type", "model", "val", "std_between_gpus", "Iteration Number"]])
    
    avg_df = pd.concat(exp_frames, axis=0, ignore_index=True)
    avg_df = avg_df.groupby(["dataset", "model_type"])


    for (dataset, model_type), group in avg_df:
        group["speedup"] = 0.0

        for it in group["Iteration Number"].unique():
            group_iter = group[group["Iteration Number"] == it]

            comparison = group_iter.loc[group_iter["model"] == "simple", "val"].values[0]

            group.loc[group["Iteration Number"] == it, "speedup"] = comparison / group_iter["val"]


        print("=" * 40)
        print("Dataset: ", dataset, "\tModel Type: ", model_type)
        group_avg_per_model = group.groupby("model")["speedup"].mean()
        group_std_between_GPUS_per_model = group.groupby("model")["std_between_gpus"].mean()
        group_std_val_per_iteration = group.groupby("model")["val"].std()

        print("Average speedup per model:")
        print(group_avg_per_model)

        print("Average Standard deviation (across iterations) between GPUs per model:")
        print(group_std_between_GPUS_per_model)

        print("Standard deviation per iteration per model:")
        print(group_std_val_per_iteration)
        print()


def plot_e2e_gpu_time(dirs: List[str]):
    avg_df = pd.DataFrame(columns=["average gpu time", "model", "gpu number"])
    for _dir in dirs:
        data = read_data(_dir)
        for folder in list_folders(_dir):
            gpu_num = int(folder)  
            e2e_gpu_times = []
            for layer in list_files(f"{_dir}/{folder}"):
                if layer == "e2e.csv":
                    continue
                df = pd.read_csv(f"{_dir}/{folder}/{layer}")
                df = df[df["iteration"] > 3]
                e2e_gpu_times.append(df["gpu processing time"].mean())
            avg_df.loc[len(avg_df)] = {"average gpu time": sum(e2e_gpu_times) / len(e2e_gpu_times), "model": data["label"], "gpu number": gpu_num}
    

    fig = px.bar(avg_df, x="gpu number", y="average gpu time", color="model", labels={"gpu number": "GPU", "average gpu time": "Average GPU execution time (ms)"}, barmode="group", text="average gpu time")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=True)
    update_fig_to_theme(fig)

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/e2e_gpu_time.png")

def plot_section_time_by_GPU(dirs: List[str]):
    columns_of_interest = ["Before Metadata and 1st Data", "Metadata and 1st Data", "Before 2nd Data", "2nd Data", "After 2nd Data"]

    total_df = None
    for _dir in dirs:
        data = read_data(_dir)
        frames = []
        for gpu in list_folders(_dir):
            frames_GPU = []
            for layer in list_files(f"{_dir}/{gpu}"):
                if layer == "e2e.csv":
                    continue
                df = pd.read_csv(f"{_dir}/{gpu}/{layer}")
                df = df[df["iteration"] > 3]
                df["layer"] = layer.split(".")[0]
                frames_GPU.append(df)

            gpu_df = pd.concat(frames_GPU, axis=0, ignore_index=True)
            gpu_df["gpu"] = gpu



            assert len(gpu_df) == (len(list_files(f"{_dir}/{gpu}")) - 1 ) * 96, f"Expected {len(list_files(f'{_dir}/{gpu}')) - 1} * 96 = {(len(list_files(f'{_dir}/{gpu}')) - 1) * 96} but got {len(gpu_df)}"

            frames.append(gpu_df)
        
        experiment_df = pd.concat(frames, axis=0, ignore_index=True)
        experiment_df = experiment_df.sort_values(["iteration", "gpu"])
        experiment_df.drop(["total number of bytes sent", "total number of bytes recv"], axis=1, inplace=True)

        grouped_df = experiment_df.groupby(["iteration", "layer"])[columns_of_interest]

        means = grouped_df.mean()
        stds = grouped_df.std()

        if total_df is None:
            total_df = pd.DataFrame(columns=["model", "section", "std", "mean", "section_idx", "layer"])

        means_per_layer = means.groupby("layer")
        stds_per_layer = stds.groupby("layer")

        for (layer, group), (layer2, group2) in zip(means_per_layer, stds_per_layer):
            assert layer == layer2, f"Expected {layer} but got {layer2}"

            iters_mean = group.mean()
            mean_diff_sq = ((group - iters_mean) ** 2).mean()
            combined_std = np.sqrt(group2.pow(2).mean() + mean_diff_sq)


            for col in columns_of_interest:
                total_df.loc[len(total_df)] = {"model": model_mapping(data["expert_manager"]), "section": col, "std": combined_std.loc[col], "mean": iters_mean.loc[col], "section_idx": columns_of_interest.index(col), "layer": layer}

    total_df = total_df[total_df["layer"].isin(["moe_l1", "moe_l11", "moe_l1_decode", "moe_l11_decode"])]
    total_df["layer"] = total_df["layer"].apply(lambda x: "_".join(x.split("_")[1:])[1:])

    total_df["layer_sort_key"] = total_df["layer"].apply(lambda x: int(x) if x.isdigit() else int(x.split("_")[0]) + 20)
        



    
    total_df = total_df.sort_values(["section_idx",  "layer_sort_key", "model"])
    fig = px.bar(total_df,
            x="section",
            y="mean",
            color="model",
            barmode="group",
            error_y="std",
            # text="std",
            facet_col="layer",
            facet_col_wrap=2,
            labels={"mean": "Avg Latency (ms)", "model": "Model", "section": "Code Section", "layer": "Layer"},
    )

    fig.for_each_xaxis(lambda x: x.update(title=dict(font=dict(size=16)), color="#000000", showgrid=True, gridcolor="#e5e5e5", titlefont=dict(size=28), tickfont=dict(size=26)))
    fig.for_each_yaxis(lambda x: x.update(title=dict(font=dict(size=16)), color="#000000", showgrid=True, gridcolor="#e5e5e5", titlefont=dict(size=28), tickfont=dict(size=26)))

    fig.update_layout(
    annotations=[
        dict(
            x=anno["x"],
            y=anno["y"],
            text=anno["text"],  # Keep the same facet title
            font=dict(size=28),  # Increase font size
            showarrow=False,
            xref="paper",
            yref="paper"
        )
        for anno in fig.layout.annotations  # Loop through all subplot titles
    ],
    legend_title=dict(font=dict(size=28)),  # Increase font size of the legend title
    legend=dict(font=dict(size=26))  # Increase font size of legend labels
    )

    fig.show()
    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/section_time_mean_std-per_GPU.pdf", format="pdf")
    total_df.to_csv(f"{OUTPUT_DIR}/section_time_mean_std-per_GPU.csv", index=False)


def plot_e2e_per_layer(dirs: [str]):
    frames = []
    for _dir in dirs:
        if _dir == "":
            return None
        data = read_data(_dir)
        print(f"Model: {model_mapping(data['label'])} - Model Type: {model_type_mapping(data['model_type'])} - Name: {data['name']if 'name' in data else data['experiment_name']}")
        frames_same_layer = defaultdict(list)
        for gpu in list_folders(_dir):
            for layer in list_files(os.path.join(_dir, gpu)):
                if layer == "e2e.csv":
                    continue
                df = pd.read_csv(os.path.join(_dir, gpu, layer))
                df = df[df["iteration"] > 3]
                df.rename(columns={"latency (ms)": f"GPU {gpu}"}, inplace=True)
                df = df[["iteration", f"GPU {gpu}"]]
                frames_same_layer[layer.split(".")[0]].append(df)
        
        f_df = []
        for layer, dfs in frames_same_layer.items():
            l_df = dfs[0]
            for df in dfs[1:]:
                l_df = l_df.merge(df, on="iteration", how="outer")
            
            l_df["val"] = l_df.iloc[:, l_df.columns != "iteration"].mean(axis=1)
            l_df["layer"] = layer
            f_df.append(l_df)
        
        f_df = pd.concat([d[["iteration", "val", "layer"]] for d in f_df], axis=0, ignore_index=True)
        f_df["model"] = model_mapping(data["label"])
        # f_df["model_type"] = model_type_mapping(data["model_type"])
        frames.append(f_df)
    
    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df[df["layer"].isin(["moe_l1", "moe_l11", "moe_l1_decode", "moe_l11_decode"])]
    df["layer"] = df["layer"].apply(lambda x: "_".join(x.split("_")[1:])[1:])

        
    fig = px.line(
        df,
        x="iteration",
        y="val",
        color="model",
        color_discrete_map={"DeepSpeed": "gray", "Sliced": "red", "MegaBlocks": "blue"},
        labels={"model": "Model", "iteration": "Iteration Number", "val": "Latency (ms)", "layer": "Layer"},
        facet_col="layer",
        facet_col_wrap=2
    )

    create_dir_if_needed()
    fig.write_image(f"{OUTPUT_DIR}/e2e_layer.pdf", format="pdf")

    

def plot_average_time_per_component(dirs: List[str]):
    def plot_avg(df, title, name="_profiling_avg_time.png"):
        #plot average
        x_axis = [col for col in df.columns if col not in ["iteration", "total number of tokens recv", "total number of tokens sent", "total number of bytes sent", "total number of bytes recv", "is_decoder", "layer", "name", "gpu"]]

        grouped_df = df.groupby("name")[x_axis].mean().reset_index()
        long_df = pd.melt(grouped_df, id_vars="name", var_name="metric", value_name="mean_value")
        def firt_digit(x):
            strs = str(x).split("_")
            for s in strs:
                if s.isdigit():
                    return int(s)
        long_df["sort_key"] = long_df["name"].apply(lambda x: firt_digit(x))
        long_df = long_df.sort_values("sort_key")

        fig = px.bar(long_df, x="metric", y="mean_value", color="name", labels={"metric": "Code Segment", "mean_value": "Average Time (ms)"}, text="mean_value", title=title)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='auto')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide', showlegend=True)
        update_fig_to_theme(fig)

        create_dir_if_needed()
        fig.write_image(f"{OUTPUT_DIR}/{name}")
    

    final_df = None
    for _dir in dirs:
        data = read_data(_dir)
        for gpu in list_folders(_dir):
            for layer_file in list_files(f"{_dir}/{gpu}"):
                if layer_file == "e2e.csv":
                    continue

                df = pd.read_csv(f"{_dir}/{gpu}/{layer_file}")
                df = df[df["iteration"] > 3]

                is_decoder_layer = "decode" in layer_file
                layer = int(layer_file.split(".")[0].split("_")[1][1:])

                df["layer"] = layer
                df["is_decoder"] = is_decoder_layer
                df["name"] = data["name"]
                df["gpu"] = int(gpu)
                if final_df is None:
                    final_df = df
                else:
                    final_df = pd.concat([final_df, df])
    
    all_layers = final_df[["layer", "is_decoder"]].drop_duplicates()
    for layer, is_decoder in zip(all_layers["layer"], all_layers["is_decoder"]):
        print(f"Plotting for layer {layer} {'decoder' if is_decoder else 'encoder'}")   
        plot_avg(final_df[(final_df["layer"] == layer) & (final_df["is_decoder"] == is_decoder)], f"Average time per component for {'decoder' if is_decoder else 'encoder'} layer {layer}", f"_profiling_avg_time_layer_{layer}_{'decoder' if is_decoder else 'encoder'}.png")
    

    for  gpu in np.sort(final_df["gpu"].unique()):
        print(f"Plotting for gpu {gpu}")
        plot_avg(final_df[final_df["gpu"] == gpu], f"Average time per component for GPU {gpu}", f"_profiling_avg_time_gpu_{gpu}.png")
    
    plot_avg(final_df, "Average time per component", "_profiling_avg_time.png")


def plot_imbalance(_dir: str):
    def get_cv(x):
        return np.std(x, axis=0) / np.mean(x, axis=0)

    data_dir = os.path.join(_dir, "0")
    final_df = []
    for layer in list_files(data_dir):
        if layer == "e2e.csv" or layer.endswith("_decode.csv"):
            continue
        df = pd.read_csv(os.path.join(data_dir, layer))
        df = df[df["iteration"] > 3]
        df = df[["iteration", "tokens per expert"]]
        df["tokens per expert"] = df["tokens per expert"].apply(ast.literal_eval)
        df["layer"] = layer.split(".")[0]
        df["cv"] = df["tokens per expert"].apply(get_cv)
        max_cv_per_layer_rows = df[df["cv"] == df["cv"].max()].copy()
        min_cv_per_layer_rows = df[df["cv"] == df["cv"].min()].copy()
        max_cv_per_layer_rows["type"] = "max"
        min_cv_per_layer_rows["type"] = "min"
        final_df.append(max_cv_per_layer_rows)
        final_df.append(min_cv_per_layer_rows)
    
    final_df = pd.concat(final_df, axis=0, ignore_index=True)
    create_dir_if_needed()
    final_df.to_csv(f"{OUTPUT_DIR}/imbalance.csv", index=False)



parser = argparse.ArgumentParser()
parser.add_argument("-t", "--type", default="average_speedup", type=str)
parser.add_argument("-r", "--rest", type=str)
parser.add_argument("--evolution_type", type=str)
parser.add_argument("--evolution_label", type=str)


args = parser.parse_args()

plotting_type = args.type
if plotting_type == "e2e":
    rest = args.rest.split(" ")
    plot_e2e(rest)
    plot_e2e_per_layer(rest)
elif plotting_type == "evolution":
    # plot_average_speedup_evolution(args.metric, args.evolution_values, args.evolution_objects, args.comparisons, args.dirs, args.caching)
    rest = args.rest.split(" ")
    plot_average_speedup_evolution_2(rest, args.evolution_type, args.evolution_label)
elif plotting_type == "profile":
    rest = args.rest.split(" ")
    plot_section_time_by_GPU(rest)

elif plotting_type == "dataset":
    rest = args.rest.split(" ")
    plot_speedup_across_dataset(rest)

elif plotting_type == "save_merge_csv-evolution":
    rest = args.rest.split(" ")
    save_merge_csv_evolution(rest, args.evolution_type, args.evolution_label)

elif plotting_type == "save_merge_csv-e2e":
    rest = args.rest.split(" ")
    save_merge_csv_e2e(rest)

elif plotting_type == "save_merge_csv-time-section":
    rest = args.rest.split(" ")
    save_merge_csv_time_section(rest)
elif plotting_type == "plot-imbalance":
    rest = args.rest.split(" ")
    rest = [r for r in rest if r != ""]
    assert len(rest) == 1, "Only 1 directory is allowed for plotting imbalance"
    plot_imbalance(rest[0])

else:
    print("No plotting type of that name")