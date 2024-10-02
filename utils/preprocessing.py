import os
import slideio
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import shutil
from xml.dom import minidom
from skimage.draw import polygon
from tifffile import imread

def preprocess_camelyon():
    data_directory = "/vol/data/histo_datasets/CAMELYON/CAMELYON17/"
    mask_list = os.listdir(data_directory + "masks/")
    for m in tqdm(mask_list):
        slide = slideio.open_slide(data_directory + "masks/" + m)
        image_slide = slideio.open_slide(data_directory + "images/" + m[:-9] + ".tif")
        scene = slide.get_scene(0)
        image_scene = image_slide.get_scene(0)

        dim0 = int(np.ceil(scene.size[0] / 1024))
        dim1 = int(np.ceil(scene.size[1] / 1024))
        resolutions = np.ceil(np.log2(max(dim0,dim1)))
        for r in range(int(resolutions) + 1):
            res = 2**r * 1024
            dim0 = int(np.ceil(scene.size[0] / res))
            dim1 = int(np.ceil(scene.size[1] / res))
            last_dim = (int(scene.size[0] % res), int(scene.size[1] % res))

            if last_dim[0] == 0 and last_dim[1] == 0:
                last_dim = (res, res)
            elif last_dim[0] == 0:
                last_dim = (res, last_dim[1])
            elif last_dim[1] == 0:  
                last_dim = (last_dim[0], res)

            for i in range(dim0):
                for j in range(dim1):
                    if i == dim0-1 and j == dim1-1:
                        width = last_dim[0]
                        height = last_dim[1]
                    elif i == dim0-1:
                        width = last_dim[0]
                        height = res
                    elif j == dim1-1:
                        width = res
                        height = last_dim[1]
                    else:
                        width = res
                        height = res
                    mask = scene.read_block((i*res,j*res, width, height), (width // (2**r), height // (2**r)))
                    mask = np.where(mask == 2, 1, 0).astype(np.uint8)
                    if (np.max(mask) == 1):
                        image = image_scene.read_block((i*res,j*res, width, height), (width // (2**r), height // (2**r)))
                        # Save image and mask
                        # Save image
                        Image.fromarray(image).save(data_directory + "images_patches/" + m[:-9] + "_{}_{}_{}_{}_{}_{}.png".format(i*res,j*res,width, height, width // (2**r),  height // (2**r)))
                        # Save mask
                        Image.fromarray(mask).save(data_directory + "masks_patches/" + m[:-9] + "_{}_{}_{}_{}_{}_{}.png".format(i*res,j*res,width, height, width // (2**r),  height // (2**r)))
        
def preprocess_conic(data_directory="/vol/data/histo_datasets/CoNIC/"):
    images = np.load(data_directory+"images.npy")
    masks = np.load(data_directory+"labels.npy")
    for i in tqdm(range(len(images))):
        mask = masks[i,:,:,0]
        if np.unique(mask).shape[0] > 1:
            mask = mask.astype(np.int32)
            Image.fromarray(images[i]).save(data_directory + "images_png/" + str(i).zfill(4) + ".png")
            Image.fromarray(mask).save(data_directory + "labels_png/" + str(i).zfill(4) + ".png")

def preprocess_cpm(data_directory = "/vol/data/histo_datasets/CPM_15_and_17/"):
    dir_list = ["cpm15/", "cpm17/test/", "cpm17/train/"]
    for d in dir_list:
        mask_list = os.listdir(data_directory + d + "Labels/")
        for i in range(len(mask_list)):
            mask = loadmat(data_directory + d + "Labels/" + mask_list[i])["inst_map"]
            mask = mask.astype(np.int32)
            os.makedirs(data_directory + d + "Labels_png/", exist_ok=True)
            Image.fromarray(mask).save(data_directory + d + "Labels_png/" + mask_list[i][:-4] + ".png")

def preprocess_crag(data_directory):
    datasets = ["train2017", "val2017"]
    for d in datasets:
        os.makedirs(data_directory + "cell_CRAG/" + d + "/labels/", exist_ok=True)
        coco = COCO(data_directory+'cell_CRAG/annotations/instances_' + d + '.json')
        cat_ids = coco.getCatIds()
        for image_id in range(len(coco.imgs)):
            img = coco.imgs[image_id+1]
            if "aug" in img['file_name']:
                continue  
            anns_ids = coco.getAnnIds(imgIds=image_id+1, catIds=cat_ids, iscrowd=None)
            anns = coco.loadAnns(anns_ids)
            mask = coco.annToMask(anns[0])
            for i in range(len(anns)):
                mask += (i+2)*coco.annToMask(anns[i])
            mask = mask.astype(np.uint8)
            Image.fromarray(mask).save(data_directory + "cell_CRAG/" + d + "/labels/" + img['file_name'])

def preprocess_icia2018():
    data_directory = "/home/ubuntu/thesis/data/ICIA2018/ICIAR2018_BACH_Challenge/WSI/"
    mask_list = ["A"+str(i+1).zfill(2)+".npy" for i in range(10)]
    for m in tqdm(mask_list):
        scene = np.load(data_directory + m).transpose()
        image_slide = slideio.open_slide(data_directory + m[:-4] + ".svs")
        image_scene = image_slide.get_scene(0)

        dim0 = int(np.ceil(scene.shape[0] / 1024))
        dim1 = int(np.ceil(scene.shape[1] / 1024))
        resolutions = np.ceil(np.log2(max(dim0,dim1)))
        for r in range(int(resolutions) + 1):
            res = 2**r * 1024
            dim0 = int(np.ceil(scene.shape[0] / res))
            dim1 = int(np.ceil(scene.shape[1] / res))
            last_dim = (int(scene.shape[0] % res), int(scene.shape[1] % res))

            if last_dim[0] == 0 and last_dim[1] == 0:
                last_dim = (res, res)
            elif last_dim[0] == 0:
                last_dim = (res, last_dim[1])
            elif last_dim[1] == 0:  
                last_dim = (last_dim[0], res)

            for i in range(dim0):
                for j in range(dim1):
                    if i == dim0-1 and j == dim1-1:
                        width = last_dim[0]
                        height = last_dim[1]
                    elif i == dim0-1:
                        width = last_dim[0]
                        height = res
                    elif j == dim1-1:
                        width = res
                        height = last_dim[1]
                    else:
                        width = res
                        height = res
                    #mask = scene.read_block((i*res,j*res, width, height), (width // (2**r), height // (2**r)))
                    mask = scene[i*res:i*res+width:2**r, j*res:j*res+height:2**r]
                    #mask = np.where(mask == 2, 1, 0).astype(np.uint8)
                    if (np.max(mask) > 0):
                        image = image_scene.read_block((i*res,j*res, width, height), (width // (2**r), height // (2**r)))
                        # Save image and mask
                        # Save image
                        Image.fromarray(image).save(data_directory + "images_patches/" + m[:-9] + "_{}_{}_{}_{}_{}_{}.png".format(i*res,j*res,width, height, width // (2**r),  height // (2**r)))
                        # Save mask
                        Image.fromarray(mask).save(data_directory + "masks_patches/" + m[:-9] + "_{}_{}_{}_{}_{}_{}.png".format(i*res,j*res,width, height, width // (2**r),  height // (2**r)))
        
def preprocess_kumar(data_directory):
    dirs = ["train/", "test_same/", "test_diff/"]
    for d in dirs:
        dir_list = os.listdir(data_directory + d + "Labels/")
        for i in dir_list:
            masks = loadmat(data_directory + d + "Labels/" + i)["inst_map"]
            os.makedirs(data_directory + d + "Labels_png_new/", exist_ok=True)
            Image.fromarray(masks).save(data_directory + d + "Labels_png_new/" + i[:-4] + ".png")

def preprocess_monusac(data_directory = '/home/ubuntu/thesis/data/MoNuSAC/'):
    image_source = data_directory + 'MoNuSAC_images_and_annotations/'
    mask_source = data_directory + 'MoNuSAC_masks/'
    image_destination = data_directory + 'images/'
    mask_destination = data_directory + 'masks/'
    image_list = os.listdir(image_source)

    # Training data
    for i in image_list:
        subimage_list = os.listdir(image_source + i)
        for j in subimage_list:
            if j.endswith('.tif'):
                target_image_file = image_destination + j
                shutil.copyfile(image_source + i + '/' + j, target_image_file)
                types = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"]
                for t in types:
                    if os.path.exists(mask_source + i + '/' + j[:-4] + '/' + t):
                        mask_list = os.listdir(mask_source + i + '/' + j[:-4] + '/' + t)
                        for m in mask_list:
                            target_mask_file = mask_destination + j[:-4] + '_' + t + '.png'
                            slide = slideio.open_slide(mask_source + i + '/' + j[:-4] + '/' + t + '/' + m, "GDAL")
                            scene = slide.get_scene(0)
                            mask = scene.read_block((0,0, scene.size[0], scene.size[1]))
                            Image.fromarray(mask.astype(np.uint8)).save(target_mask_file)
                            
    # Testing data
    source = data_directory + 'MoNuSAC_Testing_Color_Coded_Masks/'
    image_list = os.listdir(source)
    for i in tqdm(image_list):
        subimage_list = os.listdir(source + i)
        for j in subimage_list:
            if j.endswith('.png'):
                target_image_file = image_destination + j
                shutil.copyfile(source + i + '/' + j, target_image_file)
            else:
                target_mask_file = mask_destination + j[:-17] + '.png'
                mask_array = np.array(Image.open(source + i + '/' + j))
                binary_arr = np.zeros((mask_array.shape[0], mask_array.shape[1]))
                for x in range(mask_array.shape[0]):
                    for y in range(mask_array.shape[1]):
                        if mask_array[x, y, 0] == 255 or mask_array[x, y, 1] == 255 or mask_array[x, y, 2] == 255:
                            binary_arr[x][y] = 1
                Image.fromarray(binary_arr.astype(np.uint8)).save(target_mask_file)

def preprocess_monuseg(data_directory = "/home/ubuntu/thesis/data/MoNuSeg/MoNuSeg 2018 Training Data/"):
    def he_to_binary_mask(filename):
        im_file = data_directory + "Tissue Images/" + filename + '.tif'
        xml_file = data_directory + "Annotations/" + filename + '.xml'

        # Parse the XML file
        xDoc = minidom.parse(xml_file)
        Regions = xDoc.getElementsByTagName('Region')

        xy = []
        for regioni in range(Regions.length):
            Region = Regions.item(regioni)
            verticies = Region.getElementsByTagName('Vertex')
            xy_region = np.zeros((verticies.length, 2))
            for vertexi in range(verticies.length):
                x = float(verticies.item(vertexi).getAttribute('X'))
                y = float(verticies.item(vertexi).getAttribute('Y'))
                xy_region[vertexi] = [x, y]
            xy.append(xy_region)

        arr = imread(im_file)
        # Get image information
        im_info = {
            'Height': arr.shape[0],
            'Width': arr.shape[1]
        }
        binary_mask = np.zeros((im_info['Height'], im_info['Width']))
        color_mask = np.zeros((im_info['Height'], im_info['Width'], 3))

        for zz, region in enumerate(xy):
            print(f'Processing object # {zz + 1}')
            smaller_x = region[:, 0]
            smaller_y = region[:, 1]

            # Create binary and color masks
            polygon_mask = polygon(smaller_y, smaller_x, (im_info['Height'], im_info['Width']))
            binary_mask[polygon_mask] += zz + 1
            color_mask[polygon_mask] += np.random.rand(3)

        return binary_mask, color_mask

    image_list = os.listdir(data_directory + "Tissue Images/")
    for i in image_list:
        binary_mask, color_mask = he_to_binary_mask(i[:-4])
        values = np.unique(binary_mask)
        masks = np.zeros(binary_mask.shape)
        for k in range(len(values)):
            masks = np.where(binary_mask == values[k], k, masks)
        os.makedirs(data_directory + "Masks_new/", exist_ok=True)
        Image.fromarray(masks.astype(np.int32)).save(data_directory + "Masks_new/" + i[:-4] + ".png")

def preprocess_nuclick(data_directory = "/vol/data/histo_datasets/NuClick/IHC_nuclick/IHC/"):
    splits = ["Train", "Validation"]
    for s in splits:
        masks_list = os.listdir(data_directory + "masks/" + s)
        for m in masks_list:
            mask = np.load(data_directory + "masks/" + s + "/" + m)
            mask = mask.astype(np.uint8)
            if np.unique(mask).shape[0] > 1: 
                Image.fromarray(mask).save(data_directory + "masks_png/" + s + "/" + m[:-4] + ".png")

def preprocess_pannuke(data_source="/vol/data/histo_datasets/PanNuke/"):
    folds = [("Fold 1/", "fold1/"), ("Fold 2/", "fold2/"), ("Fold 3/", "fold3/")]
    os.makedirs(data_source + "images_png/", exist_ok=True)
    os.makedirs(data_source + "masks_png/", exist_ok=True)
    counter = 0
    for f in folds:
        images = np.load(data_source + f[0] + "images/" + f[1] + "images.npy")
        masks = np.load(data_source + f[0] + "masks/" + f[1] + "masks.npy")
        for i in tqdm(range(images.shape[0])):
            output = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.int32)
            k = 0
            for j in range(masks.shape[3]-1):
                values = np.unique(masks[i, :, :, j])
                for v in values:
                    if v != 0:
                        output[masks[i, :, :, j] == v] = k
                        k += 1
            if np.unique(output).shape[0] > 1:
                Image.fromarray(images[i].astype(np.uint8)).save(data_source + "images_png/" + str(counter).zfill(4) + ".png")
                Image.fromarray(output).save(data_source + "masks_png/" + str(counter).zfill(4) + ".png")
                counter += 1
        del images
        del masks

def preprocess_segpc(data_source = "/vol/data/histo_datasets/SegPC/TCIA_SegPC_dataset/"):
    splits = ["train", "validation"]
    for s in splits:
        images_list = os.listdir(data_source + s + "/x/")
        masks_list = os.listdir(data_source + s + "/y/")
        os.makedirs(data_source + s + "/masks_png/", exist_ok=True)
        for i in images_list:
            short_masks_list = [m for m in masks_list if i[:-4] == m[:-6]]
            j = 0
            for m in short_masks_list:
                if j == 0:
                    mask = np.array(Image.open(data_source + s + "/y/" + m))
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                    mask = np.where(mask == 20, 1, mask)
                    mask = np.where(mask == 40, 2, mask)
                else:
                    additional_mask = np.array(Image.open(data_source + s + "/y/" + m))
                    if len(additional_mask.shape) == 3:
                        additional_mask = additional_mask[:, :, 0]
                    mask = np.where(additional_mask == 20, mask + 2*j -1, mask)
                    mask = np.where(additional_mask == 40, mask + 2*j, mask)
                j += 1
            Image.fromarray(mask).save(data_source + s + "/masks_png/" + m[:-6] + ".png")

def preprocess_wsss4luad(data_directory = '/home/ubuntu/thesis/data/WSSS4LUAD/2.validation/'):
    source_directory = data_directory + 'mask/'
    output_directory = data_directory + 'masks_relabeled/'
    os.makedirs(output_directory, exist_ok=True)
    dir_list = os.listdir(source_directory)
    for d in dir_list:
        arr = np.array(Image.open(source_directory + d))
        arr[arr == 3] = 255
        arr = arr + 1
        Image.fromarray(arr).save(output_directory + d)

preprocess_pannuke()