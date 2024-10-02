import torch
from ..modeling.model import SamHI
from ..data_processing.data_utils import PromptProcessing
from ..data_processing.dataset import prepare_data
from pathlib import Path
import numpy as np
from PIL import Image
import os 
import pandas as pd
from monai.metrics import compute_iou, compute_dice
import json
import random
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, SamPredictor
import cv2
import torch.nn.functional as F
from isegm.inference.evaluation import evaluate_sample
from isegm.inference.predictors import get_predictor
from isegm.inference import utils
from .inference import Inference
from segment_anything.utils.transforms import ResizeLongestSide
from skimage.transform import resize as sk_resize

class Evaluation(Inference):
    def __init__(self, config):
        self.config = config
        self.inference_config = config["inference_config"]
        self.load_config(self.inference_config)
        self.load_eval_config()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        if not os.path.exists(Path(self.model_dir + self.model_name + "_" + self.time + "_eval_config.json")):
            os.mknod(Path(self.model_dir + self.model_name + "_" + self.time + "_eval_config.json"))
        self.initialize_model_evaluation()

    def load_eval_config(self):
        self.data_config = self.config["data_config"]
        self.eval_config = self.config["eval_config"]
        self.prompt_config = self.config["prompt_config"]
        self.interactive_or_auto = self.eval_config["interactive_or_auto"]
        self.interactive_model = self.eval_config["interactive_model"]
        self.auto_model = self.eval_config["auto_model"]
        self.time = self.eval_config["time"]
        self.seed = self.eval_config["seed"]
        self.max_nr_of_points = self.eval_config["max_nr_of_points"]
        self.batch_size = self.eval_config["batch_size"]
        self.prompt_batch_size = self.eval_config.get("prompt_batch_size", 1)
        self.nr_of_interactive_points_choices = self.eval_config.get("nr_of_interactive_points_choices", [0, 1, 2, 3, 4, 5])
        self.prompt_choices = self.eval_config.get("prompt_choices", ["points", "boxes"])
        self.dataset_choices = self.eval_config["dataset_choices"]
        self.mask_threshold = self.eval_config.get("mask_threshold", 0.0)
        self.model_type = self.eval_config["model_type"]
        self.base_model = self.eval_config["base_model"]

    def initialize_model_evaluation(self):
        if self.interactive_or_auto == "interactive":
            if self.interactive_model == "simpleclick":
                self.initialize_simpleclick()
            elif self.interactive_model == "sam":
                self.initialize_sam()
            else:
                self.initialize_model()
        else:
            if self.auto_model == "cellvit":
                self.initialize_cellvit()
            elif self.auto_model == "sam":
                self.initialize_sam()
            else:
                self.initialize_cellvit()
                self.initialize_model()
        
    def initialize_simpleclick(self):
        checkpoint_path = self.model_dir + "SimpleClick/cocolvis_vit_base.pth"
        self.model = utils.load_is_model(checkpoint_path, self.device, False)
        self.model.eval()
        self.predictor = get_predictor(self.model, 'NoBRS', self.device, prob_thresh=0.49, zoom_in_params=None)
        self.max_iou_thr = 1.0
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).cuda()
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).cuda()
        self.data_config["image_encoder_size"] = 448
        self.prompt_batch_size = 1

    def initialize_sam(self):
        model_config = {
            "model_type": self.model_type,
            "base_model": self.base_model,
            "lora_rank": 1,
            "lora_layer": [-1],
            "model_mode": "inference",
            "model_dir": self.model_dir,
        }
        model = SamHI({"model_config": model_config}).to(self.device)
        model.eval()
        self.model = model
        self.predictor = SamPredictor(model.model)
    
    @torch.no_grad()
    def evaluate(self):
        if self.interactive_or_auto == "auto":
            self.evaluate_auto()
        else:
            self.initialize_eval_dict(self.max_nr_of_points)
            self.predicted_masks = []
            self.target_masks = []
            for dataset_name in self.dataset_choices:
                self.data_config["datasets"] = [dataset_name]
                dataset, dataloader = prepare_data(self.data_config)
                for (i1, p1) in enumerate(self.prompt_choices):
                    self.prompt_config["prompt_type"] = p1
                    for (i2, p2) in enumerate(self.nr_of_interactive_points_choices): 
                        nr_of_interactive_points = p2
                        for (idx, (data, target, nr, (image_name, mask_name), (left, upper, right, lower), (new_h, new_w)), p_tune) in tqdm(dataloader):
                            self.update_eval_dict_first_part(dataset_name, nr_of_interactive_points, idx, image_name, mask_name, left, upper, right, lower) 
                            data, target = data.to(self.device), target.to(self.device)
                            prompts, target, target_nr = PromptProcessing.get_prompts_and_targets(nr,  target, self.device, self.prompt_config)
                            if self.interactive_model == "simpleclick":
                                self.evaluate_simpleclick(data, target, target_nr, nr_of_interactive_points + 1, new_h, new_w)
                            else:
                                output, _, image_embeddings = self.model.forward(data, prompts, p_tune=p_tune)
                                for i in range(nr_of_interactive_points): 
                                    prompts= PromptProcessing.refine_prompts(nr, target, prompts, output, self.device,self.prompt_batch_size)
                                    output, _, _ = self.model.forward(data, prompts, image_embeddings=image_embeddings, p_tune=p_tune)
                                output = output > self.mask_threshold
                                self.predicted_masks.extend([m for m in output.squeeze().cpu().numpy()])
                                self.target_masks.extend([m for m in target.cpu().numpy()])
                                self.update_eval_dict_second_part( prompts, target_nr, output, target)
        self.save_eval_results(self.predicted_masks, self.target_masks)


    def evaluate_simpleclick(self, data, target, target_nr, max_clicks, new_h, new_w):
        for i in range(len(data)):
            image = data[i]
            image = image[:, :new_h[i], :new_w[i]]
            image = image * self.pixel_std + self.pixel_mean
            image = F.pad(image, (0, 448 - new_w[i], 0, 448 - new_h[i]))
            image = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            gt = target[i].cpu().numpy()
            clicks_list, ious, probs = evaluate_sample(image, gt, self.predictor, self.max_iou_thr, max_clicks=max_clicks)
            prompt = {
                "point_coords": torch.zeros(1, max_clicks, 2),
                "point_labels": torch.zeros(1, max_clicks)
            }
            for (c, click) in enumerate(clicks_list):
                prompt["point_coords"][0, c, 0] = click.coords[1]
                prompt["point_coords"][0, c, 1] = click.coords[0]
                prompt["point_labels"][0,c] = int(click.is_positive)
            output = (probs > 0.49)
            self.predicted_masks.extend([output])
            self.target_masks.extend([target[i].cpu().numpy()])
            self.update_eval_dict_second_part([prompt], [target_nr[i]], torch.tensor(output).unsqueeze(0).unsqueeze(1), target[i].cpu().unsqueeze(0))

    def evaluate_auto(self):
        transform = ResizeLongestSide(1024)
        self.eval_dict = {
            "Dataset": [],
            "Index": [],
            "Image name": [],
            "Mask name": [],
            "Left": [],
            "Upper": [],
            "Right": [],
            "Lower": [],
            "IoU": [],
            "Dice": [],
        }
        self.predicted_masks = []
        self.target_masks = []
        pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).cuda()
        pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).cuda()
        for dataset_name in self.dataset_choices:
            self.data_config["datasets"] = [dataset_name]
            dataset, dataloader = prepare_data(self.data_config)
            for (idx, (data, target, _,(image_name, mask_name), (left, upper, right, lower), (new_h, new_w)), p_tune) in tqdm(dataloader):
                with torch.no_grad():  
                    data, target = data.to(self.device), target.to(self.device)
                    for (i,d) in enumerate(data):
                        img = np.array(Image.open(image_name[i]).convert("RGB"))
                        img = transform.apply_image(img)
                        h, w = img.shape[0], img.shape[1]
                        # img = d[:, :new_h[i], :new_w[i]]
                        # img = img * pixel_std + pixel_mean 
                        # img = img.cpu().numpy().astype(np.uint8)   
                        # img = img.transpose((1, 2, 0))     
                        if self.auto_model=="cellvit":
                            mask = self.segment_automatically_cellvit(img)
                        elif self.auto_model=="sam":
                            mask, _ = self.segment_automatically_sam(img)
                            mask = np.pad(mask,((0, 1024-mask.shape[0]), (0, 1024-mask.shape[1])))
                        else:
                            mask,_ = self.segment_automatically(img)
                            mask = np.pad(mask,((0, 1024-mask.shape[0]), (0, 1024-mask.shape[1])))
                        mask = np.where(mask == 0, 0, 1)
                        gt = np.array(Image.open(mask_name[i]))
                        gt = np.where(gt == 0, 0, 1).astype(np.uint8)
                        gt = sk_resize(gt, (h,w), preserve_range=True, order = 0)
                        gt = np.pad(gt,((0, 1024-gt.shape[0]), (0, 1024-gt.shape[1])))
                        # gt = torch.where(target[i].unsqueeze(0).unsqueeze(1).cpu() == 0, 0, 1)
                        self.predicted_masks.extend([mask])
                        self.target_masks.extend([gt])
                        self.eval_dict["Dataset"].extend([dataset_name])
                        self.eval_dict["Index"].extend([idx[i].item()])
                        self.eval_dict["Image name"].extend([image_name[i]])
                        self.eval_dict["Mask name"].extend([mask_name[i]])
                        self.eval_dict["Left"].extend([left[i].item()])
                        self.eval_dict["Upper"].extend([upper[i].item()])
                        self.eval_dict["Right"].extend([right[i].item()])
                        self.eval_dict["Lower"].extend([lower[i].item()])
                        self.eval_dict["IoU"].extend([compute_iou(torch.tensor(mask[np.newaxis, np.newaxis, :, :]), torch.tensor(gt[np.newaxis, np.newaxis, :, :])).item()])
                        self.eval_dict["Dice"].extend([compute_dice(torch.tensor(mask[np.newaxis, np.newaxis, :, :]), torch.tensor(gt[np.newaxis, np.newaxis, :, :])).item()])
        #self.save_eval_results(self.predicted_masks, self.target_masks)

    def initialize_eval_dict(self, max_nr_of_prompts=4):
        eval_dict = {
            "Dataset": [],
            "Prompt type": [],
            "Nr of interactive points": [],
            "Index": [],
            "Image name": [],
            "Mask name": [],
            "Left": [],
            "Upper": [],
            "Right": [],
            "Lower": [],
            "Target nr": [],
            "IoU": [],
            "Dice": [],
        }
        for i in range(4):
            eval_dict["Box_coord_" + str(i+1)] = []
        for i in range(max_nr_of_prompts):
            eval_dict["Point_coord_" + str(2*i+1)] = []
            eval_dict["Point_coord_" + str(2*i+2)] = []
            eval_dict["Point_label_" + str(i+1)] = []
        self.eval_dict = eval_dict

    def update_eval_dict_first_part(self, dataset, nr_of_interactive_points, idx, image_name, mask_name, left, upper, right, lower):
        self.eval_dict["Dataset"].extend([dataset] * self.batch_size * self.prompt_batch_size)
        self.eval_dict["Nr of interactive points"].extend([nr_of_interactive_points] * self.batch_size * self.prompt_batch_size)
        self.eval_dict["Prompt type"].extend([self.prompt_config["prompt_type"]] * self.batch_size * self.prompt_batch_size)
        self.eval_dict["Index"].extend([i.item() for i in idx for j in range(self.prompt_batch_size)])
        self.eval_dict["Image name"].extend([i for i in image_name for j in range(self.prompt_batch_size)])
        self.eval_dict["Mask name"].extend([m for m in mask_name for j in range(self.prompt_batch_size)])
        self.eval_dict["Left"].extend([l.item() for l in left for j in range(self.prompt_batch_size)])
        self.eval_dict["Upper"].extend([u.item() for u in upper for j in range(self.prompt_batch_size)])
        self.eval_dict["Right"].extend([r.item() for r in right for j in range(self.prompt_batch_size)])
        self.eval_dict["Lower"].extend([l.item() for l in lower for j in range(self.prompt_batch_size)])


    def update_eval_dict_second_part(self, prompts, target_nr, output, target):
        self.eval_dict["Target nr"].extend(target_nr)
        for (i, p) in enumerate(prompts):
            if "point_coords" in p.keys():
                points = p["point_coords"].cpu().numpy()
                labels = p["point_labels"].cpu().numpy()
                for j in range(self.max_nr_of_points):
                    if j < points.shape[1]:
                        self.eval_dict["Point_coord_" + str(2*j+1)].extend([pt[j,0] for pt in points])
                        self.eval_dict["Point_coord_" + str(2*j+2)].extend([pt[j,1] for pt in points])
                        self.eval_dict["Point_label_" + str(j+1)].extend([l[j] for l in labels])
                    else:
                        self.eval_dict["Point_coord_" + str(2*j+1)].extend([0 for p in points])
                        self.eval_dict["Point_coord_" + str(2*j+2)].extend([0 for p in points])
                        self.eval_dict["Point_label_" + str(j+1)].extend([0 for l in labels])
            else:
                for j in range(self.max_nr_of_points):
                    self.eval_dict["Point_coord_" + str(2*j+1)].extend([0 for p in range(self.prompt_batch_size)])
                    self.eval_dict["Point_coord_" + str(2*j+2)].extend([0 for p in range(self.prompt_batch_size)])
                    self.eval_dict["Point_label_" + str(j+1)].extend([0 for l in range(self.prompt_batch_size)])
            
            if "boxes" in p.keys():
                boxes = p["boxes"].cpu().numpy()
                for i in range(4):   
                    self.eval_dict["Box_coord_" + str(i+1)].extend([b[i] for b in boxes])
            else:
                for i in range(4):
                    self.eval_dict["Box_coord_" + str(i+1)].extend([0 for b in range(self.prompt_batch_size)])
        self.eval_dict["IoU"].extend(compute_iou(output, target.unsqueeze(1)).squeeze(1).cpu().numpy().tolist())
        self.eval_dict["Dice"].extend(compute_dice(output, target.unsqueeze(1)).squeeze(1).cpu().numpy().tolist())

    def save_eval_results(self, predicted_masks, target_masks):
        os.makedirs(self.model_dir + self.model_name + "_" + self.time + "_eval_results", exist_ok=True)
        for (i,p) in enumerate(predicted_masks):
            p = np.where(p == 1, 255, 0).astype(np.uint8)
            p = Image.fromarray(p)
            p.save(self.model_dir + self.model_name + "_" + self.time + "_eval_results/" + str(i) +  "_pred_mask.png")

        for (i,t) in enumerate(target_masks):
            t = np.where(t == 1, 255, 0).astype(np.uint8)
            t = Image.fromarray(t)
            t.save(self.model_dir + self.model_name + "_" + self.time + "_eval_results/" + str(i) + "_target_mask.png")

        df = pd.DataFrame(self.eval_dict)
        df.to_csv(Path(self.model_dir + self.model_name + "_" + self.time + "_eval_results.csv"))
        with open(Path(self.model_dir + self.model_name + "_" + self.time + "_eval_config.json"), "w") as file:
            self.device = str(self.device)
            json.dump(self.config, file)

    def segment_automatically_sam(self, image):
        mask_generator = SamAutomaticMaskGenerator(self.model.model)
        masks = mask_generator.generate(image)
        sorted_masks = sorted(masks, key=lambda x: x['stability_score'], reverse=False)
        # Get the masks and prompts
        prel_masks = np.zeros(image.shape[:2])
        prel_points = []
        for (i, m) in enumerate(sorted_masks):
            prel_masks[m["segmentation"]] = i
            prel_points.append(m['point_coords'])
        # filter out masks that are covered up by other masks
        prompts = []
        masks = np.zeros(image.shape[:2])
        for (i, v) in enumerate(np.unique(prel_masks)):
            masks = np.where(prel_masks == v, i, masks)
            prompts.append({"point_coords":torch.tensor(prel_points[int(v)]).to(self.device), "point_labels":torch.tensor([1]).to(self.device)})
        return masks, prompts
    
    def segment_automatically_cellvit(self, image):
        instance_types = self.detect_cellvit(image)
        base = np.zeros((1024,1024))
        masks = []
        for k in instance_types[0].keys():
            masks.append(cv2.drawContours(base, [instance_types[0][k]['contour'][:, np.newaxis,:]], 0, 1, cv2.FILLED))
        try:
            mask = np.array(masks).max(axis=0)
        except:
            mask = np.zeros((1024,1024))
        return mask
    