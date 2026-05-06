import json
import os

# ===== 原始配置 =====
base_config = {
    "prefix": "2",
    "dataset": "cub",
    "data_path": "/data/wfq/dataset/cub",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": True,
    "shuffle": True,
    "init_cls": 10,
    "increment": 10,
    "model_name": "dlora",
    "net_type": "sip",
    "embd_dim": 768,
    "num_heads": 12,
    "total_sessions": 20,
    "seed": [0,1993,1996],
    "EPSILON": 1e-8,
    "init_epoch": 20,
    "optim": "adam",
    "init_lr": 0.0005,
    "init_lr_decay": 0.1,
    "init_weight_decay": 0.0,
    "epochs": 20,
    "lrate": 0.0005,
    "lrate_decay": 0.1,
    "batch_size": 128,
    "weight_decay": 0.0,
    "rank": 10,
    "lamb": 0.95,
    "lame": 1.0,
    "num_workers": 16,
    "disloss": "spec",
    "layer": {
        "mode": "prefix",
        "k": 2
    }
}

# ===== 输出目录 =====
out_dir = "./"
os.makedirs(out_dir, exist_ok=True)

configs = []

# prefix 实验
for k in [2,3,4,5,6,7,8,9,10,11]:
    configs.append(("prefix", k))

# suffix 实验
for k in [2,3,4,5,6,7,8,9,10,11]:
    configs.append(("suffix", k))

# random 实验
for k in [2,3,4,5,6,7,8,9,10,11]:
    configs.append(("random", k))

# ===== 生成 JSON =====
for mode, k in configs:

    cfg = base_config.copy()
    cfg["layer"] = {
        "mode": mode,
        "k": k
    }

    # 日志前缀
    cfg["prefix"] = f"{mode}_{k}"

    filename = f"{mode}_{k}.json"
    path = os.path.join(out_dir, filename)

    with open(path, "w") as f:
        json.dump(cfg, f, indent=4)

print("JSON files generated in:", out_dir)