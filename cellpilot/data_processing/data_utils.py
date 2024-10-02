import numpy as np
import slideio
import torch
from PIL import Image
import torch.nn.functional as F
from skimage.transform import resize as sk_resize
import random
from segment_anything.utils.transforms import ResizeLongestSide
from scipy.ndimage import label, convolve
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore
import tifffile
import albumentations as A

class DataProcessing:
    
    def preprocess(image_name, mask_name, pixel_mean, pixel_std, wsi_image, image_encoder_size, mask_augmentation_tries, data_augmentations, threshold_connected_components, coords=None):
        image, gt, patch_coordinates, resized_size = DataProcessing.patch(image_name, mask_name, wsi_image, image_encoder_size=image_encoder_size, coords=coords)
        if data_augmentations != ["NoOp"]:
            image, gt = DataProcessing.augment(image, gt, data_augmentations, mask_augmentation_tries)
        image = DataProcessing.preprocess_image(image, pixel_mean, pixel_std, image_encoder_size=image_encoder_size)
        image = image.float()
        gt = DataProcessing.preprocess_gt(gt, image_encoder_size=image_encoder_size)
        gt = DataProcessing.connected_component_analysis(gt, threshold_connected_components)
        nr_labels = torch.max(gt)
        return image, gt, nr_labels, (image_name, mask_name), patch_coordinates, resized_size
    
    def augment(image, gt, data_augmentations, mask_augmentation_tries=5):
        image_augmentation_dict = {
            "AdvancedBlur": A.AdvancedBlur(),
            "Blur": A.Blur(),
            "GaussianBlur": A.GaussianBlur(),
            "ZoomBlur": A.ZoomBlur(),
            "CLAHE": A.CLAHE(),
            "Emboss": A.Emboss(),
            "GaussNoise": A.GaussNoise(),
            "IsoNoise": A.ISONoise(),
            "ImageCompression": A.ImageCompression(),
            "Posterize": A.Posterize(),
            "RingingOvershoot": A.RingingOvershoot(),
            "Sharpen": A.Sharpen(),
            "ToGray": A.ToGray(),
            "Downscale": A.Downscale(scale_range=(0.5, 0.9), p=0.5),
            "ChannelShuffle": A.ChannelShuffle(),
            "ChromaticAberration": A.ChromaticAberration(),
            "ColorJitter": A.ColorJitter(),
            "HueSaturationValue": A.HueSaturationValue(),
            "MultiplicativeNoise": A.MultiplicativeNoise(),
            "PlanckianJitter": A.PlanckianJitter(),
            "RGBShift": A.RGBShift(),
            "RandomBrightnessContrast": A.RandomBrightnessContrast(),
            "RandomGamma": A.RandomGamma(),
            "RandomToneCurve": A.RandomToneCurve(),
            "FancyPCA": A.FancyPCA(),
        }
        mask_augmentation_dict = {
            "Affine": A.Affine(),
            "CropNonEmptyMaskIfExists": A.CropNonEmptyMaskIfExists(512, 512, p=0.5),
            "ElasticTransform": A.ElasticTransform(),
            "GridDistortion": A.GridDistortion(),
            "OpticalDistortion": A.OpticalDistortion(),
            "RandomCrop": A.RandomCrop(512, 512, p=0.5),
            "RandomGridShuffle": A.RandomGridShuffle(),
            "RandomResizedCrop": A.RandomResizedCrop(size=(1024, 1024),p=0.5),
            "Rotate": A.Rotate(),
            "ShiftScaleRotate": A.ShiftScaleRotate(),
            "CropAndPad": A.CropAndPad(px=10, p=0.5),
            "D4": A.D4(p=0.5),
            "PadIfNeeded": A.PadIfNeeded(p=0.5),
            "Perspective": A.Perspective(),
            "RandomScale": A.RandomScale(),
        }
        image_augmentations = [image_augmentation_dict[da] for da in data_augmentations if da in image_augmentation_dict]
        mask_augmentations = [mask_augmentation_dict[da] for da in data_augmentations if da in mask_augmentation_dict]
        image_transform = A.Compose(image_augmentations)
        mask_transform = A.Compose(mask_augmentations)
        transformed = image_transform(image=image)
        image = transformed["image"]
        for i in range(mask_augmentation_tries):
            transformed = mask_transform(image=image, mask=gt)
            if np.unique(transformed["mask"]).shape[0] > 1:
                image = transformed["image"]
                gt = transformed["mask"]
                break
        return image, gt

    def patch(image_name, gt_name, wsi_image=False, image_encoder_size=1024, coords=None):
        if wsi_image:
            image = slideio.open_slide(image_name)
            image_scene = image.get_scene(0)
            if gt_name.endswith(".npy"):
                gt = np.load(gt_name).transpose()
            else:
                gt = slideio.open_slide(gt_name)
                gt_scene = gt.get_scene(0)
            h, w = image_scene.size
        else: 
            if image_name.endswith(".tiff") or image_name.endswith(".tif"):
                image = tifffile.imread(image_name)
            else:
                image = Image.open(image_name)
            if gt_name.endswith(".tiff") or gt_name.endswith(".tif"):
                gt = tifffile.imread(gt_name)
            else:
                gt = Image.open(gt_name)
            image = np.array(image)
            gt = np.array(gt)
            h, w = image.shape[:2]

        def random_patches(h, w):
            if coords is not None:
                left, right, upper, lower = coords
                return left, upper, right, lower
            else:
                left = random.randint(0, max(0, h - image_encoder_size))
                upper = random.randint(0, max(0, w - image_encoder_size))
                right = random.randint(min(h,left + image_encoder_size), h)
                lower = random.randint(min(w, upper + image_encoder_size), w)
                return left, upper, right, lower
        
        def grid_patches(i, h, w):
            left = i % ((h // image_encoder_size) + 1) * image_encoder_size
            upper = i // ((h // image_encoder_size) + 1) * image_encoder_size
            right = min(h, left + image_encoder_size)
            lower = min(w, upper + image_encoder_size)
            return left, upper, right, lower
        
        nr_of_random_samples = 10
        i = 0
        while True:
            if i < nr_of_random_samples:
                left, upper, right, lower = random_patches(h, w)
            else:
                left, upper, right, lower = grid_patches(i - nr_of_random_samples, h, w)
            i += 1

            new_h, new_w = ResizeLongestSide.get_preprocess_shape(right - left, lower - upper, image_encoder_size)
            if wsi_image:
                image_resized = image_scene.read_block((left, upper, right-left, lower-upper), (new_h, new_w))
                if gt_name.endswith(".npy"):
                    gt_cropped = gt[left:right, upper:lower].astype(np.uint8)
                    gt_resized = sk_resize(gt_cropped, (new_h,new_w), preserve_range=True, order = 0)
                else:
                    gt_resized = gt_scene.read_block((left, upper, right-left, lower-upper), (new_h, new_w))
            else:
                image_cropped = image[left:right, upper:lower]
                try:
                    if np.max(image_cropped) > 255:
                        image_cropped = (255/np.max(image_cropped)) * image_cropped
                except:
                    pass
                image_resized = np.array(resize(to_pil_image(image_cropped.astype(np.uint8)), (new_h, new_w)))
                gt_cropped = gt[left:right, upper:lower].astype(np.uint8)
                gt_resized = sk_resize(gt_cropped, (new_h,new_w), preserve_range=True, order = 0)

            if np.unique(gt_resized).shape[0] > 1:
                return image_resized, gt_resized, (left, upper, right, lower), (new_h, new_w)

    def preprocess_image(x, pixel_mean, pixel_std, image_encoder_size=1024):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        if len(x.shape) == 2:
            x = np.repeat(x[:, :, np.newaxis], 3, axis=2)
        if x.shape[2] == 4:
            x = x[:, :, :3]
        x = x.transpose((2,0,1))
        x = torch.tensor(x)
        x = (x - pixel_mean) / pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = image_encoder_size - h
        padw = image_encoder_size - w
        x = F.pad(x, (0, padw, 0, padh))     
        return x
    
    def preprocess_gt(x, image_encoder_size=1024):
        """Pad to a square input."""
        # Pad
        h, w = x.shape[-2:]
        padh = image_encoder_size - h
        padw = image_encoder_size - w
        x = torch.tensor(x)
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def connected_component_analysis(gt, threshold):
        structure = np.ones((3, 3), dtype=np.int32)
        mask_values= np.unique(gt)
        mask_values= mask_values[1:]
        counter = 0
        cca_gt = np.zeros_like(gt, dtype=np.int32)
        for v in mask_values:
            binary_gt_mask = np.where(gt == v, 1.0, 0.0)
            labeled_gt_mask, ncomponents = label(binary_gt_mask, structure)
            counts = np.bincount(labeled_gt_mask.flatten())[1:]
            j = 0
            for (i, c) in enumerate(counts):
                if c < threshold:
                    labeled_gt_mask = np.where(labeled_gt_mask == i + 1, 0, labeled_gt_mask)
                else:
                    j += 1
                    labeled_gt_mask = np.where(labeled_gt_mask == i + 1, j, labeled_gt_mask)
            labeled_gt_mask = np.where(labeled_gt_mask > 0, labeled_gt_mask+counter, 0)
            counter += j
            cca_gt += labeled_gt_mask
        cca_gt = torch.tensor(cca_gt)
        return cca_gt

    def unconnected_component_analysis(gt):
        mask_values= np.unique(gt)
        mask_values= mask_values[1:]
        uca_gt = np.zeros_like(gt, dtype=np.int32)
        for (i, v) in enumerate(mask_values):
            uca_gt = np.where(gt == v, i+1, uca_gt)
        uca_gt = torch.tensor(uca_gt)
        return uca_gt



class PromptProcessing: 

    @staticmethod
    def get_prompts_and_targets(nr, target, device, prompt_config):
        "Get prompts to be used in the model"
        prompt_batch_size = prompt_config["prompt_batch_size"]
        prompt_type = prompt_config["prompt_type"]
        nr_of_points = prompt_config["nr_of_points"]
        nr_of_pos_points = prompt_config["nr_of_positive_points"]
        bbox_shift = prompt_config["bbox_shift"]
        components = [[random.randint(1, nr[j]) for i in range(prompt_batch_size)] for j in range(len(nr))]
        targets = []
        target_nr = []
        for i in range(len(components)):
            for j in range(len(components[i])):
                targets.append(torch.where(target[i] == components[i][j], 1, 0)) 
                target_nr.append(components[i][j])
        target = torch.stack(targets, dim=0)

        if prompt_type == "both" or prompt_type == "points":
            nr_of_points_per_component = [nr_of_points for j in range(len(components))]
            nr_of_pos_points_per_component = [nr_of_pos_points for j in range(len(components))]
            prompts = PromptProcessing.get_point_prompts(target, nr, prompt_batch_size, nr_of_points_per_component, nr_of_pos_points_per_component, device)
        else:
            prompts = PromptProcessing.get_box_prompts(target, components, device, bbox_shift)
        if prompt_type == "both":
            box_prompts = PromptProcessing.get_box_prompts(target, components, device, bbox_shift)
            prompts = prompts + box_prompts
            target = torch.cat((target, target), 0)
        return prompts, target, target_nr
    
    @staticmethod
    def get_point_prompts(target, nr, prompt_batch_size, nr_of_points, nr_of_pos_points, device):
        prompts = []
        idx = 0
        for i in range(len(nr)):
            prompt = {}
            point_coords = torch.zeros(prompt_batch_size, nr_of_points[i], 2)
            point_labels = torch.ones(prompt_batch_size, nr_of_points[i]) 
            point_labels[:, nr_of_pos_points[i]:] = 0
            for j in range(prompt_batch_size):
                x_indices, y_indices  = PromptProcessing.filter_out_edge(target[idx])
                for k in range(nr_of_pos_points[i]):
                    rand_idx = random.randrange(0, len(x_indices), 1)
                    point_coords[j, k, 0] = y_indices[rand_idx]
                    point_coords[j, k, 1] = x_indices[rand_idx]
                x_indices, y_indices = PromptProcessing.filter_out_edge(1-target[idx])
                for k in range(nr_of_points[i] - nr_of_pos_points[i]):
                    rand_idx = random.randrange(0, len(x_indices), 1)
                    point_coords[j, k + nr_of_pos_points[i], 0] = y_indices[rand_idx]
                    point_coords[j, k + nr_of_pos_points[i], 1] = x_indices[rand_idx]
                idx += 1
            point_coords, point_labels = point_coords.to(device), point_labels.to(device)
            prompt.update({
                "point_coords": point_coords,
                "point_labels": point_labels,
            })
            prompts.append(prompt)
        return prompts

    @staticmethod
    def filter_out_edge(target):
        kernel = np.ones((3,3))
        target_np = target.cpu().numpy()
        inside = convolve(target_np, kernel, mode='constant', cval=0.0)
        if np.any(inside == 9):
            return np.where(inside == 9)
        else: 
            return np.where(target_np == 1)

    @staticmethod
    def get_box_prompts(target, components, device, bbox_shift):
        prompts = []
        idx = 0    
        for i in range(len(components)):
            prompt = {}
            bboxes = torch.zeros(len(components[i]), 4)
            for j in range(len(components[i])):
                y_indices, x_indices = torch.where(target[idx] == 1)
                x_min, x_max = torch.min(x_indices), torch.max(x_indices)
                y_min, y_max = torch.min(y_indices), torch.max(y_indices)
                # add perturbation to bounding box coordinates
                _,H, W = target.shape
                x_min = max(0, x_min - random.randint(0, bbox_shift))
                x_max = min(W, x_max + random.randint(0, bbox_shift))
                y_min = max(0, y_min - random.randint(0, bbox_shift))
                y_max = min(H, y_max + random.randint(0, bbox_shift))
                bboxes[j,0] = x_min
                bboxes[j,1] = y_min
                bboxes[j,2] = x_max
                bboxes[j,3] = y_max
                idx += 1
            bboxes = bboxes.to(device) 
            prompt["boxes"] = bboxes
            prompts.append(prompt)
        return prompts

    @staticmethod
    def postprocess_masks(masks, input_size=(1024,1024), original_size=(1024,1024)):
        masks = F.interpolate(
            masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    @staticmethod
    def refine_prompts(nr, target, previous_prompts, previous_prediction, device, prompt_batch_size):
        binary_prediction = (previous_prediction > 0).float()
        diff = target.unsqueeze(1) - binary_prediction
        pos_diff = diff > 0
        neg_diff = diff < 0
        structure = np.ones((3, 3), dtype=np.int32)
        
        for i in range(len(previous_prompts)):
            prompt_list = []
            prompt_label_list = []
            for j in range(prompt_batch_size):
                conn_comp_pos, threshold = label(pos_diff[prompt_batch_size * i + j][0].cpu().numpy(), structure)
                conn_comp_neg = label(neg_diff[prompt_batch_size * i + j][0].cpu().numpy(), structure)[0]
                conn_comp = conn_comp_pos + np.where(conn_comp_neg > 0, conn_comp_neg + threshold, 0)
                component_size = np.bincount(conn_comp.flatten())[1:]
                if "point_coords" in previous_prompts[i]:
                    prompt_list.append(previous_prompts[i]["point_coords"][j])
                    prompt_label_list.append(previous_prompts[i]["point_labels"][j])
                if component_size.size == 0:
                    max_indices = [0] 
                else:
                    max_indices = [np.argmax(component_size) + 1]
                for m in max_indices:
                    target_m = torch.tensor(np.where(conn_comp == m, 1, 0))
                    if m == 0:
                        label_m = torch.tensor([0]).float().to(device)
                    else:
                        label_m = torch.tensor([(m - 1 < threshold)]).float().to(device)
                    prompts = PromptProcessing.get_point_prompts(target_m.unsqueeze(0), [1], 1, [1], [1], device)
                    if "point_coords" in previous_prompts[i]:
                        prompt_list[j] = torch.cat((prompt_list[j], prompts[0]["point_coords"][0]), 0)
                        prompt_label_list[j] = torch.cat((prompt_label_list[j], label_m), 0)
                    else: 
                        prompt_list.append(prompts[0]["point_coords"][0])
                        prompt_label_list.append(label_m)
            prompt_stack = torch.stack(prompt_list, dim=0)
            prompt_label_stack = torch.stack(prompt_label_list, dim=0)
            previous_prompts[i]["point_coords"] = prompt_stack
            previous_prompts[i]["point_labels"] = prompt_label_stack
        return previous_prompts
