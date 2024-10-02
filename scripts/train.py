import argparse
import wandb
import datetime
import os
from cellpilot.modeling.model import SamHI
from cellpilot.data_processing.dataset import prepare_data
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch

time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define arguments
parser = argparse.ArgumentParser()

parser.add_argument("--cluster", type=str, choices=["denbi", "helmholtz", "custom"], default="denbi")
parser.add_argument("--num_workers", type=int, default=5)

# W&B parameters
parser.add_argument("--project_name", type=str, default="SAMHI")
parser.add_argument("--entity", type=str, default="philippresearch")

# Model Info
parser.add_argument("--model_type", type=str, choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
parser.add_argument("--model_dir", type=str, default="/vol/data/models/")  
parser.add_argument("--base_model", type=str, default="sam_vit_b_01ec64.pth")
parser.add_argument("--loss", type=str, choices=["diceCE", "diceFocal", "dice", "generalized_dice", "generalized_diceFocal", "tversky"], default="diceCE")
parser.add_argument("--compile_model", type=bool, default=False)
parser.add_argument("--save_model", type=bool, default=False)
parser.add_argument("--resume_training", type=bool, default=False)
parser.add_argument("--resume_ckpt", type=str, default="model-czpfnbsk:v0")

# Finetuning approach
parser.add_argument("--freeze", nargs='+', choices=["prompt", "mask", "image", ""], default=["prompt", "image"])
parser.add_argument("--image_encoder_size", type=int, default=1024)
parser.add_argument("--p_tuning", type=bool, default=False)

# Dataset type and location
datasets = ["BCSS", "CAMELYON", "CellSeg", "CoCaHis", "CoNIC", "CPM", "CRAG", "CryoNuSeg", "GlaS", "ICIA2018",
            "Janowczyk", "KPI", "Kumar", "MoNuSAC", "MoNuSeg", "NuClick", "PAIP2023", "PanNuke", "SegPath", "SegPC",
            "TIGER", "TNBC", "WSSS4LUAD"]
parser.add_argument("--datasets", nargs='+', choices=datasets, default=["Janowczyk"])
parser.add_argument("--use_holdout_testset", type=bool, default=False)
parser.add_argument("--test_datasets", nargs='+', choices=datasets, default=["CPM"])
parser.add_argument("--data_directory", type=str, default="/vol/data/histo_datasets/")
augmentations = ["AdvancedBlur", "Blur", "GaussianBlur", "ZoomBlur", "CLAHE", "Emboss", "GaussNoise", "IsoNoise",
                "ImageCompression", "Posterize", "RingingOvershoot", "Sharpen", "ToGray", "Downscale",
                "ChannelShuffle", "ChromaticAberration", "ColorJitter","HueSaturationValue", "MultiplicativeNoise",
                "PlanckianJitter", "RGBShift", "RandomBrightnessContrast", "RandomGamma","RandomToneCurve",
                "FancyPCA",
                "Affine", "CropNonEmptyMaskIfExists", "ElasticTransform", "GridDistortion", "OpticalDistortion",
                "RandomCrop", "RandomGridShuffle", "RandomResizedCrop", "RandomResizedCrop", "Rotate",
                "ShiftScaleRotate", "CropAndPad", "D4", "PadIfNeeded", "Perspective", "RandomScale",
                "NoOp"]
parser.add_argument("--data_augmentations", nargs='+', default=["NoOp"])
parser.add_argument("--mask_augmentation_tries", type=int, default=5)

parser.add_argument("--threshold_connected_components", type=int, default=2)

#Training parameters
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--shuffle", type=bool, default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument("--mask_threshold", type=float, default=0.0)
parser.add_argument("--bbox_shift", type=int, default=10)
parser.add_argument("--run_directory", type=str, default="/vol/data/runs/")

# Prompt parameters
parser.add_argument("--random_prompt_type", type=bool, default=False)
parser.add_argument("--prompt_type", type=str, choices=["points", "boxes", "both"], default="both")
parser.add_argument("--prompt_batch_size", type=str, default=8)

# Data split
parser.add_argument("--data_split", type=bool, default=True)
parser.add_argument("--train_split", type=float, default=0.6)
parser.add_argument("--val_split", type=float, default=0.2)
parser.add_argument("--test_split", type=float, default=0.2)

# DataLoader parameters
parser.add_argument("--drop_last", type=bool, default=True)

# LoRA parameters
parser.add_argument("--lora_rank", type=int, default=1)
parser.add_argument("--lora_layer", nargs='+', default=None) #["-1"]

# Interactive parameters
parser.add_argument("--random_mode", type=bool, default=False)
parser.add_argument("--mode", type=str, choices=["random", "interactive"], default="interactive")
parser.add_argument("--random_nr_of_interactive_points", type=bool, default=False)
parser.add_argument("--max_nr_of_interactive_points", type=int, default=1)
parser.add_argument("--nr_of_interactive_points", type=int, default=0)

# Point parameters
parser.add_argument("--random_nr_of_points", type=bool, default=False)
parser.add_argument("--max_nr_of_points", type=int, default=1)
parser.add_argument("--nr_of_initial_points", type=int, default=1)

# Positive point parameters
parser.add_argument("--only_positive_points", type=bool, default=False)
parser.add_argument("--random_nr_of_positive_points", type=bool, default=False)
parser.add_argument("--max_nr_of_positive_points", type=int, default=1)
parser.add_argument("--nr_of_initial_positive_points", type=int, default=1)

parser.add_argument("--display_name", type=str, default="")

args = parser.parse_args()


# Choose display_name
if args.display_name == "":
    display_name = f"{time}"
else:
    display_name = args.display_name
   
initial_config = {
    "compile_model": args.compile_model,
    "save_model": args.save_model,
    "display_name": display_name,
    "epochs": args.epochs,
    "time": time,
    "mask_threshold": args.mask_threshold,
    "accumulate_grad_batches": args.accumulate_grad_batches, 
    "seed": args.seed,
    "run_directory": args.run_directory,
    "project_name": args.project_name,
    "entity": args.entity,
    "resume_training": args.resume_training,
    "model_dir": args.model_dir,
    "resume_ckpt": args.resume_ckpt,
}
model_config = {
    "base_model": args.base_model,
    "model_type": args.model_type,
    "lora_layer": args.lora_layer,
    "lora_rank": args.lora_rank,
    "model_mode": "train",
    "model_dir": args.model_dir,
    "p_tuning": args.p_tuning,
}
train_config = {
    "loss": args.loss,
    "learning_rate": args.lr,
    "freeze": args.freeze,
    "mode": args.mode,
    "nr_of_interactive_points": args.nr_of_interactive_points,
    "compile_model": args.compile_model,
    "mask_threshold": args.mask_threshold,
    "prompt_batch_size": args.prompt_batch_size,
    "prompt_type": args.prompt_type,
    "batch_size": args.batch_size,

}
data_config = {
    "datasets": args.datasets,
    "data_directory": args.data_directory,
    "cluster": args.cluster,
    "image_encoder_size": args.image_encoder_size,
    "batch_size": args.batch_size,
    "drop_last": args.drop_last,
    "num_workers": args.num_workers,
    "use_holdout_testset": args.use_holdout_testset,
    "test_datasets": args.test_datasets,
    "data_augmentations": args.data_augmentations,
    "mask_augmentation_tries": args.mask_augmentation_tries,
    "threshold_connected_components": args.threshold_connected_components,
    "data_split": args.data_split,
    "train_split": args.train_split,
    "val_split": args.val_split,
    "test_split": args.test_split,
    "shuffle": args.shuffle,
    "seed": args.seed,
}
prompt_config = {
    "prompt_batch_size": args.prompt_batch_size, 
    "prompt_type": args.prompt_type,
    "nr_of_points": args.nr_of_initial_points, 
    "nr_of_positive_points": args.nr_of_initial_positive_points,
    "bbox_shift": args.bbox_shift,
}
random_prompt_config = {
    "random_mode": args.random_mode,
    "random_prompt_type": args.random_prompt_type,
    "random_nr_of_points": args.random_nr_of_points,
    "random_nr_of_interactive_points": args.random_nr_of_interactive_points,
    "random_nr_of_positive_points": args.random_nr_of_positive_points,
    "max_nr_of_interactive_points": args.max_nr_of_interactive_points,
    "max_nr_of_points": args.max_nr_of_points,
    "max_nr_of_positive_points": args.max_nr_of_positive_points,  
    "only_positive_points": args.only_positive_points,
}

config = {
    "initial_config": initial_config,
    "model_config": model_config,
    "training_config": train_config,
    "data_config": data_config,
    "prompt_config": prompt_config,
    "random_prompt_config": random_prompt_config,
}

print("CONFIG: ", config)

### TRAINING
compile_model = initial_config["compile_model"]
save_model = initial_config["save_model"]
seed = initial_config["seed"]
run_directory = initial_config["run_directory"]
project_name = initial_config["project_name"]
entity = initial_config["entity"]
resume_training = initial_config["resume_training"]
model_dir = initial_config["model_dir"]
resume_ckpt = initial_config["resume_ckpt"]

torch.set_float32_matmul_precision("high")
pl.seed_everything(seed, workers=True)
model = SamHI(config)
if compile_model:
    model = torch.compile(model)
if save_model:
    wandb_logger = WandbLogger(log_model="all", project=project_name, entity=entity, name=display_name, config=config, save_dir=run_directory)
else:
    wandb_logger = WandbLogger(log_model=False, project=project_name, entity=entity, name=display_name, config=config, save_dir=run_directory)
trainer = pl.Trainer(logger=wandb_logger, 
                     accelerator="gpu",
                     devices=1,
                     deterministic=False,
                     fast_dev_run=False,
                     max_epochs=initial_config["epochs"],
                     accumulate_grad_batches=initial_config["accumulate_grad_batches"])
train_loader, val_loader, test_loader = prepare_data(data_config)
if resume_training:
    model_path = os.path.join(model_dir, resume_ckpt)
    os.makedirs(model_path, exist_ok=True)
    api = wandb.Api()
    artifact = api.artifact("philippresearch/SAMHI/" + resume_ckpt, type="model")
    artifact_dir = artifact.download(root=model_path)
    trainer.fit(model, train_loader, val_loader, ckpt_path=os.path.join(model_path, "model.ckpt"))
else:
    trainer.fit(model, train_loader, val_loader)
pl.seed_everything(seed, workers=True)

random_prompt_config = {
    "random_mode": False,
    "random_prompt_type": False,
    "random_nr_of_points": False,
    "random_nr_of_interactive_points": False,
    "random_nr_of_positive_points": False,
    "only_positive_points": True,
}
model.random_prompt_config = random_prompt_config
prompt_config = {
    "prompt_batch_size": 1, 
    "nr_of_points": 1,
    "prompt_type": "both",
    "nr_of_positive_points": 1,
    "bbox_shift": 10,
}
model.prompt_config = prompt_config
model.prompt_type = "both"
model.nr_of_interactive_points = 0
model.prompt_batch_size = 1
trainer.test(model, dataloaders=test_loader)