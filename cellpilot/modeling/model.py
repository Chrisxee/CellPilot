from .LoRA_Sam import LoRA_Sam
from segment_anything import sam_model_registry
import schedulefree
from ..data_processing.data_utils import PromptProcessing
import monai
import random
import torch
import numpy as np
import wandb
from monai.metrics import compute_iou, compute_dice
import os

class SamHI (LoRA_Sam):
    def __init__(self, config):
        self.config = config
        self.model_config = config["model_config"]
        self.load_config()
        model_path = os.path.join(self.model_dir, self.base_model)
        super().__init__(sam_model_registry[self.model_type](checkpoint=model_path), self.lora_rank, self.lora_layer, self.p_tuning)
        self.model = self.sam
        if self.model_mode == "train":
            self.training_config = config["training_config"]
            self.prompt_config = config["prompt_config"]
            self.random_prompt_config = config["random_prompt_config"]
            self.training_init()
        

    def load_config(self):
        self.base_model = self.model_config["base_model"]
        self.model_type = self.model_config["model_type"]
        self.lora_layer = self.model_config["lora_layer"]
        self.lora_rank = self.model_config["lora_rank"]
        self.model_mode = self.model_config["model_mode"] 
        self.model_dir = self.model_config["model_dir"] 
        self.p_tuning = self.model_config.get("p_tuning", False)    
    
    def training_init(self):
        self.load_training_config()
        loss_dict = {
            "diceCE": monai.losses.DiceCELoss(sigmoid=True),
            "diceFocal": monai.losses.DiceFocalLoss(sigmoid=True),
            "dice": monai.losses.DiceLoss(sigmoid=True),
            "generalized_dice": monai.losses.GeneralizedDiceLoss(sigmoid=True),
            "generalized_diceFocal": monai.losses.GeneralizedDiceFocalLoss(sigmoid=True),
            "tversky": monai.losses.TverskyLoss(sigmoid=True),         
        }
        self.loss = loss_dict[self.loss]
        self.save_hyperparameters()
        self.freeze_parameters()


    def load_training_config(self):
        self.loss = self.training_config["loss"]
        self.freeze = self.training_config["freeze"]
        self.learning_rate = self.training_config["learning_rate"]
        self.batch_size = self.training_config["batch_size"]
        self.prompt_batch_size = self.training_config["prompt_batch_size"]
        self.mode = self.training_config["mode"]
        self.prompt_type = self.training_config["prompt_type"]
        self.nr_of_interactive_points = self.training_config["nr_of_interactive_points"]
        self.compile_model = self.training_config["compile_model"]
        self.mask_threshold = self.training_config["mask_threshold"]
    
    def randomization(self):
        if self.random_prompt_config["random_mode"]:
            self.mode = random.choice(["random", "interactive"])
        if self.random_prompt_config["random_prompt_type"]:
            self.prompt_type = random.choice(["points", "boxes"])
        if self.random_prompt_config["random_nr_of_interactive_points"]:
            self.nr_of_interactive_points = random.randint(1, self.random_prompt_config["max_nr_of_interactive_points"])
        if self.random_prompt_config["random_nr_of_points"]:
            self.prompt_config["nr_of_points"] = random.randint(1, self.random_prompt_config["max_nr_of_points"])
        if self.random_prompt_config["random_nr_of_positive_points"]:
            self.prompt_config["nr_of_positive_points"] = random.randint(1, self.random_prompt_config["max_nr_of_positive_points"])
        if self.random_prompt_config["only_positive_points"]:
            self.prompt_config["nr_of_positive_points"] = self.prompt_config["nr_of_points"]


    def freeze_parameters(self):
        if "image" in self.freeze:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if "prompt" in self.freeze:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if "mask" in self.freeze:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, data, prompts, image_embeddings=None, p_tune=None):
        return super().forward(data, prompts, False, 1024, image_embeddings, p_tune)
    
    def configure_optimizers(self):
        optimizer = schedulefree.AdamWScheduleFree(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate)
        return optimizer  

    def working_step(self, batch):
        self.randomization()
        idx, (data, target, nr, _, _, size), p_tune = batch 
        prompts, target, _ = PromptProcessing.get_prompts_and_targets(nr, target, self.device, self.prompt_config) 
        if self.mode == "interactive":
            with torch.no_grad():
                output, low_res_output, image_embeddings = self.forward(data, prompts, p_tune=p_tune)
                if self.prompt_type == "both":
                    box_output, _, _ = self.forward(data, prompts[int(len(prompts)/2):], image_embeddings, p_tune=p_tune)
                    output = torch.cat([output, box_output], dim=0)
                for i in range(self.nr_of_interactive_points): 
                    prompts = PromptProcessing.refine_prompts(nr, target, prompts, output, self.device, self.prompt_batch_size)
                    output, low_res_output, _ = self.forward(data, prompts, image_embeddings, p_tune=p_tune) 
                    if self.prompt_type == "both":
                        box_output, _, _ = self.forward(data, prompts[int(len(prompts)/2):], image_embeddings, p_tune=p_tune)
                        output = torch.cat([output, box_output], dim=0)
        output, _, image_embeddings = self.forward(data, prompts, p_tune=p_tune)
        if self.prompt_type == "both":
            box_output, _, _ = self.forward(data, prompts[int(len(prompts)/2):], image_embeddings, p_tune=p_tune)
            output = torch.cat([output, box_output], dim=0)
        return data, target, prompts, output

    #@torch.compile
    def compiled_working_step(self, batch):
        return self.working_step(batch)

    def unified_step(self, batch):
        if self.compile_model:
            return self.compiled_working_step(batch)
        else: 
            return self.working_step(batch)

    def training_step(self, batch, batch_idx):
        data, target, prompts, output = self.unified_step(batch)
        loss = self.loss(output.squeeze().unsqueeze(1), target.float().squeeze().unsqueeze(1))
        self.log("train_loss", loss, on_epoch=True, batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target, prompts, output = self.unified_step(batch)
        loss = self.loss(output.squeeze().unsqueeze(1), target.float().squeeze().unsqueeze(1))
        if batch_idx == 0:
            self.display_samples(data, target, prompts, output, "val")
        self.log("val_loss", loss, on_epoch=True, batch_size=self.batch_size)

    def display_samples(self, data, target, prompts, output, mode):
        img = []
        output = output > self.mask_threshold
        d = data.shape[0]
        for i in range(d):
            image = data[i].squeeze().cpu().numpy().transpose(1, 2, 0)
            if self.prompt_type == "both":
                image_masks = {
                    "target": {"mask_data": target[(i) * self.prompt_batch_size].squeeze().cpu().numpy()},
                    "output_box": {"mask_data": 2*output[(i) * self.prompt_batch_size].squeeze().cpu().numpy()},
                    "output_point": {"mask_data": 3*output[(d + i) * self.prompt_batch_size].squeeze().cpu().numpy()}
                }
            else:
                image_masks = {
                    "target": {"mask_data": target[i * self.prompt_batch_size].squeeze().cpu().numpy()},
                    "output": {"mask_data": 2*output[i * self.prompt_batch_size].squeeze().cpu().numpy()}
                }
            k = i
            if self.prompt_type in ["points", "both"]:
                points = prompts[k]['point_coords'][0]
                point_mask = np.zeros((image.shape[0], image.shape[1]))
                for (j, point) in enumerate(points):
                    point = point.cpu().numpy()
                    point_mask[int(point[1]) - 2:int(point[1]) + 2, int(point[0]) - 2:int(point[0]) + 2] = 4 + int(prompts[i]['point_labels'][0][j].item())
                image_masks["point"] = {"mask_data": point_mask}
            if self.prompt_type == "both":
                k = d + i
            if self.prompt_type in ["boxes", "both"]: 
                boxes = prompts[k]['boxes'][0].squeeze().cpu().numpy()
                box_mask = np.zeros((image.shape[0], image.shape[1]))
                box_mask[int(boxes[1]):int(boxes[3]), int(boxes[0]):int(boxes[2])] = 6
                image_masks["box"] = {"mask_data": box_mask}
                if "point_coords" in prompts[k]:
                    points = prompts[k]['point_coords'][0]
                    point_mask = np.zeros((image.shape[0], image.shape[1]))
                    for (j, point) in enumerate(points):
                        point = point.cpu().numpy()
                        point_mask[int(point[1]) - 2:int(point[1]) + 2, int(point[0]) - 2:int(point[0]) + 2] = 7 + int(prompts[k]['point_labels'][0][j].item())
                    image_masks["box_point"] = {"mask_data": point_mask}
            img.append(wandb.Image(image, masks=image_masks))
        self.trainer.logger.experiment.log({mode + "_samples": img})

    def test_step(self, batch, batch_idx):
        data, target, prompts, output = self.unified_step(batch)
        if batch_idx == 0:
            self.display_samples(data, target, prompts, output, "test")
        output = output > self.mask_threshold
        if self.prompt_type == "both":
            point_iou = compute_iou(output[:int(len(output)/2)], target[:int(len(output)/2)].unsqueeze(1))
            point_dice = compute_dice(output[:int(len(output)/2)], target[:int(len(output)/2)].unsqueeze(1))
            box_iou = compute_iou(output[int(len(output)/2):], target[int(len(output)/2):].unsqueeze(1))
            box_dice = compute_dice(output[int(len(output)/2):], target[int(len(output)/2):].unsqueeze(1))
            self.log("test_point_iou", torch.mean(point_iou), on_epoch=True, batch_size=self.batch_size)
            self.log("test_point_dice", torch.mean(point_dice), on_epoch=True, batch_size=self.batch_size)
            self.log("test_point_iou_std", torch.std(point_iou), on_epoch=True, batch_size=self.batch_size)
            self.log("test_point_dice_std", torch.std(point_dice), on_epoch=True, batch_size=self.batch_size)
            self.log("test_box_iou", torch.mean(box_iou), on_epoch=True, batch_size=self.batch_size)
            self.log("test_box_dice", torch.mean(box_dice), on_epoch=True, batch_size=self.batch_size)
            self.log("test_box_iou_std", torch.std(box_iou), on_epoch=True, batch_size=self.batch_size)
            self.log("test_box_dice_std", torch.std(box_dice), on_epoch=True, batch_size=self.batch_size)
        else:
            iou = compute_iou(output, target.unsqueeze(1))
            dice = compute_dice(output, target.unsqueeze(1))
            self.log("test_iou", torch.mean(iou), on_epoch=True, batch_size=self.batch_size)
            self.log("test_dice", torch.mean(dice), on_epoch=True, batch_size=self.batch_size)
            self.log("test_iou_std", torch.std(iou), on_epoch=True, batch_size=self.batch_size)
            self.log("test_dice_std", torch.std(dice), on_epoch=True, batch_size=self.batch_size)
    




    # make schedulefree work
    def on_fit_start(self) -> None:
        self.optimizers().train()

    def on_predict_start(self) -> None:
        self.optimizers().eval()
        
    def on_validation_model_eval(self) -> None:
        self.model.eval()
        self.optimizers().eval()

    def on_validation_model_train(self) -> None:
        self.model.train()
        self.optimizers().train()

    def on_test_model_eval(self) -> None:
        self.model.eval()
        #self.optimizers().eval()

    def on_test_model_train(self) -> None:
        self.model.train()
        self.optimizers().train()

    def on_predict_model_eval(self) -> None:  # redundant with on_predict_start()
        self.model.eval()
        self.optimizers().eval()
