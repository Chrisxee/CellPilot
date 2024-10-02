import torch
import datetime
from cellpilot.inference.evaluation_tools import Evaluation  
import argparse

time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser()

# General Info
model_types = ["vit_b", "vit_l", "vit_h"]
parser.add_argument("--model_type", type=str, choices=model_types, default="vit_b")
parser.add_argument("--base_model", type=str, default="sam_vit_b_01ec64.pth")
parser.add_argument("--model_dir", type=str, default="/vol/data/models/")
# model_names = ["sam_vit_b_01ec64"]
parser.add_argument("--model_name", type=str, default="model-g9e0e56i:v0")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--interactive_or_auto", type=str, choices=["interactive", "auto"], default="interactive")
interactive_models = ["sam", "simpleclick", "ours"]
parser.add_argument("--interactive_model", type=str, default="ours")
auto_models = ["sam", "cellvit", "ours"]
parser.add_argument("--auto_model", type=str, choices=auto_models, default="ours")
parser.add_argument("--max_nr_of_points", type=int, default=6)

# Data Info
datasets = ["BCSS", "CAMELYON", "CellSeg", "CoCaHis", "CoNIC", "CPM", "CRAG", "CryoNuSeg", "GlaS", "ICIA2018",
            "Janowczyk", "KPI", "Kumar", "MoNuSAC", "MoNuSeg", "NuClick", "PAIP2023", "PanNuke", "SegPath", "SegPC",
            "TIGER", "TNBC", "WSSS4LUAD"]
parser.add_argument("--datasets", nargs='+', choices=datasets, default=["CPM"])
parser.add_argument("--data_dir", type=str, default="/vol/data/histo_datasets/")
parser.add_argument("--cluster", type=str, default="denbi")
parser.add_argument("--image_encoder_size", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--drop_last", type=bool, default=True)
parser.add_argument("--num_workers", type=int, default=5)

# Prompt Info
parser.add_argument("--prompt_batch_size", type=int, default=10)
parser.add_argument("--nr_of_initial_points", type=int, default=1)
parser.add_argument("--nr_of_initial_positive_points", type=int, default=1)
parser.add_argument("--bbox_shift", type=int, default=10)

# Evaluation Info
parser.add_argument("--nr_of_interactive_points_choices", nargs='+', type=int, default=[0, 1, 2, 3, 4, 5])
parser.add_argument("--prompt_choices", nargs='+', choices=["points", "boxes"], default=["points", "boxes"])


args = parser.parse_args()

inference_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_dir": args.model_dir,
    "model_name": args.model_name,
}
eval_config = {
    "interactive_or_auto": args.interactive_or_auto,
    "interactive_model": args.interactive_model,
    "auto_model": args.auto_model,
    "time": time,
    "seed": args.seed,
    "batch_size": args.batch_size,
    "prompt_batch_size": args.prompt_batch_size,
    "max_nr_of_points": args.max_nr_of_points,
    "nr_of_interactive_points_choices": args.nr_of_interactive_points_choices,
    "prompt_choices": args.prompt_choices,
    "dataset_choices": args.datasets,
    "mask_threshold": 0.0,
    "model_type": args.model_type,
    "base_model": args.base_model,
}
## Evaluate on the following datasets: CRAG (glands), MoNuSeg (cells), Cellseg (cells on blood smear)
data_config = {
    "data_directory": args.data_dir,
    "cluster": args.cluster,
    "image_encoder_size": args.image_encoder_size,
    "batch_size": args.batch_size,
    "drop_last": args.drop_last,
    "num_workers": args.num_workers,
}
prompt_config = {
    "prompt_batch_size": args.prompt_batch_size, 
    "nr_of_points": args.nr_of_initial_points, 
    "nr_of_positive_points": args.nr_of_initial_positive_points,
    "bbox_shift": args.bbox_shift,
}

config = {
    "inference_config": inference_config,
    "eval_config": eval_config,
    "data_config": data_config,
    "prompt_config": prompt_config,
}
evaluation = Evaluation(config)
evaluation.evaluate()
