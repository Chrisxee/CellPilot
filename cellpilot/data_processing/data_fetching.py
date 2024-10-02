import os
import numpy as np

class DataFetcher:
    def __init__(self, data_directory, cluster):
        self.data_directory = data_directory
        self.cluster = cluster

    def len_bcss(self):
        return len(os.listdir(os.path.join(self.data_directory, 'BCSS/0_Public-data-Amgad2019_0.25MPP', 'masks')))

    def len_camelyon(self):
        self.cam_len = [len(os.listdir(os.path.join(self.data_directory, 'CAMELYON/CAMELYON16/images/'))), 
                            len(os.listdir(os.path.join(self.data_directory, 'CAMELYON/CAMELYON17/images/')))]
        return  self.cam_len[0] + self.cam_len[1]

    def len_cellseg(self):
        self.cellseg_len = [
            len(os.listdir(os.path.join(self.data_directory, 'CellSeg/NeurIPS22-CellSeg', 'Testing/Hidden/images/'))),
            len(os.listdir(os.path.join(self.data_directory, 'CellSeg/NeurIPS22-CellSeg', 'Testing/Public/images/'))),
            len(os.listdir(os.path.join(self.data_directory, 'CellSeg/NeurIPS22-CellSeg', 'Training/images/'))),
            len(os.listdir(os.path.join(self.data_directory, 'CellSeg/NeurIPS22-CellSeg', 'Tuning/images/')))
        ]
        return sum(self.cellseg_len)

    def len_cocahis(self):
        return 82

    def len_conic(self):
        return len(os.listdir(os.path.join(self.data_directory, 'CoNIC/labels_png/')))

    def len_cpm(self):
        self.cpm_len = [len(os.listdir(os.path.join(self.data_directory, 'CPM_15_and_17/cpm15/Labels_png/'))), 
                        len(os.listdir(os.path.join(self.data_directory, 'CPM_15_and_17/cpm17/test/Labels_png/'))), 
                        len(os.listdir(os.path.join(self.data_directory, 'CPM_15_and_17/cpm17/train/Labels_png/')))]
        return self.cpm_len[0] + self.cpm_len[1] + self.cpm_len[2]  
    
    def len_crag(self):
        self.crag_len = [len(os.listdir(os.path.join(self.data_directory, 'CRAG/cell_CRAG/train2017/labels/'))),
                        len(os.listdir(os.path.join(self.data_directory, 'CRAG/cell_CRAG/val2017/labels/')))]
        return self.crag_len[0] + self.crag_len[1]
    
    def len_cryonuseg(self):
        return 30

    def len_glas(self):
        return 60 + 20 + 85

    def len_icia2018(self):
        return 10

    def len_janowczyk(self):
        return 141
    
    def len_kpi(self):
        if self.cluster != "denbi":
            data_directories = [os.path.join(self.data_directory, 'KPI/', 'Task1_patch_level/data'), os.path.join(self.data_directory, 'KPI/', 'val/Task1_patch_level/data')]
        else:
            data_directories = [os.path.join(self.data_directory, 'KPI/', 'KPIs24 Training Data/Task1_patch_level/data'), os.path.join(self.data_directory, 'KPI/', 'KPIs24 Validation Data/Task1_patch_level/data')]
        arr_length = 2
        for (i0, data_directory) in enumerate(data_directories):
            dir_list = sorted(os.listdir(data_directory))
            arr_length += len(dir_list)
            for (i1, d1) in enumerate(dir_list):
                subdir_list = sorted(os.listdir(os.path.join(data_directory, d1)))
                arr_length += len(subdir_list)
        pointer = np.zeros(arr_length, dtype=int)
        length = np.zeros(arr_length, dtype=int)
        pointer[0] = len(data_directories)
        len_last_dir_list = 0
        len_last_subdir_list = 0
        for (i0, data_directory) in enumerate(data_directories):
            dir_list = sorted(os.listdir(data_directory))
            len_last_dir_list = len(dir_list)
            pointer[pointer[i0]] = pointer[i0] + len_last_dir_list
            for (i1, d1) in enumerate(dir_list):
                subdir_list = sorted(os.listdir(os.path.join(data_directory, d1)))
                len_last_subdir_list = len(subdir_list)
                for (i2, d2) in enumerate(subdir_list):
                    file_list = os.listdir(os.path.join(data_directory, d1, d2,'img'))
                    length[pointer[pointer[i0] + i1] + i2] = len(file_list)
                    length[pointer[i0] + i1] += len(file_list)
                    length[i0] += len(file_list)
                if i1 < len(dir_list) - 1:
                    pointer[pointer[i0] + i1 + 1] = pointer[pointer[i0] + i1] + len(subdir_list)
                
            if i0 < len(data_directories) - 1:
                pointer[i0 + 1] = pointer[pointer[i0] + len_last_dir_list - 1] + len_last_subdir_list
        self.pointer = pointer
        self.length = length
        # pointer = array([ 2, 36,  6, 11, 16, 31,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0, 40, 42, 44, 46,  0,  0,  0,  0,  0,  0,  0,  0])
        # length = array([5214, 1643,  558,  607, 2263, 1786,   92,   96,  141,   86,  143, 135,  134,   71,  129,  138,  143,  187,  166,  146,  133,  111, 156,  177,  152,  147,  150,   93,  156,  218,  128,  358,  390, 313,  370,  355,  274,  299,  209,  861,  144,  130,  182,  117, 106,  103,  415,  446])
        return self.length[0] + self.length[1]

    def len_kumar(self):
        self.kumar_len = [len(os.listdir(os.path.join(self.data_directory, 'Kumar', 'train/Labels_png'))),
                            len(os.listdir(os.path.join(self.data_directory, 'Kumar', 'test_same/Labels_png'))),
                            len(os.listdir(os.path.join(self.data_directory, 'Kumar', 'test_diff/Labels_png')))]
        return self.kumar_len[0] + self.kumar_len[1] + self.kumar_len[2]
    
    def len_monusac(self):
        return len(os.listdir(os.path.join(self.data_directory, 'MoNuSAC/masks/')))

    def len_monuseg(self):
        return 37

    def len_nuclick(self):
        return 1213 + 250 + 462 + 150

    def len_paip2023(self):
        return 50 + 50 + 3 + 3

    def len_pannuke(self):
        return 2656 #7466 
    
    def len_segpath(self):
        return 10647 + 26509 + 24805 + 12273 + 14135 + 13231 + 25909 + 31178
    
    def len_segpc(self):
        return 298 + 199

    def len_tiger(self):
        return 1879

    def len_tnbc(self):
        return 7 + 3 + 5 + 8 + 4 + 3 + 3 + 4 + 6 + 4 + 3

    def len_wsss4luad(self):
        return 40
        
    def get_cpm(self, idx):
        data_directory = os.path.join(self.data_directory, 'CPM_15_and_17/')
        if idx < self.cpm_len[0]:
            mask_names = os.listdir(data_directory + 'cpm15/Labels_png/')
            image_name = data_directory + 'cpm15/Images/' + mask_names[idx][:8] + ".png"
            mask_name = data_directory + 'cpm15/Labels_png/' + mask_names[idx]
        elif idx < self.cpm_len[0] + self.cpm_len[1]:
            mask_names = os.listdir(data_directory + 'cpm17/test/Labels_png/')
            image_name = data_directory + 'cpm17/test/Images/' + mask_names[idx-self.cpm_len[0]][:8] + ".png"
            mask_name = data_directory + 'cpm17/test/Labels_png/' + mask_names[idx-self.cpm_len[0]]
        else:
            mask_names = os.listdir(data_directory + 'cpm17/train/Labels_png/')
            image_name = data_directory + 'cpm17/train/Images/' + mask_names[idx-self.cpm_len[0]-self.cpm_len[1]][:8] + ".png"
            mask_name = data_directory + 'cpm17/train/Labels_png/' + mask_names[idx-self.cpm_len[0]-self.cpm_len[1]]
        return image_name, mask_name  
    
    def get_bcss(self, idx):
        data_directory = os.path.join(self.data_directory, 'BCSS/0_Public-data-Amgad2019_0.25MPP/')
        if self.cluster != "denbi":
            mask_names = os.listdir(data_directory + 'masks/')
            image_name = data_directory + 'rgbs_colorNormalized/' + mask_names[idx]
            mask_name = data_directory + 'masks/' + mask_names[idx]
        else:
            image_names = os.listdir(data_directory + 'images/')
            image_name = data_directory + 'images/' + image_names[idx]
            mask_name = data_directory + 'masks/' + image_names[idx]
        return image_name, mask_name
        
    
    def get_camelyon(self, idx):
        if idx < self.cam_len[0]:
            data_directory = os.path.join(self.data_directory, 'CAMELYON/CAMELYON16/')
        else:
            data_directory = os.path.join(self.data_directory, 'CAMELYON/CAMELYON17/')
            idx -= self.cam_len[0]
        image_names = os.listdir(data_directory + 'images/')
        image_name = data_directory + 'images/' + image_names[idx]
        mask_name = data_directory + 'masks/' + image_names[idx][:-4] + '_mask.tif'
        return image_name, mask_name

    def get_cellseg(self, idx):
        data_directory = os.path.join(self.data_directory, 'CellSeg/NeurIPS22-CellSeg/')
        if idx < self.cellseg_len[0]:
            image_names = os.listdir(data_directory + 'Testing/Hidden/images/')
            image_name = data_directory + 'Testing/Hidden/images/' + image_names[idx]
            mask_name = data_directory + 'Testing/Hidden/osilab_seg/' + image_names[idx][:14] + '_label.tiff'
        elif idx < self.cellseg_len[0] + self.cellseg_len[1]:
            image_names = os.listdir(data_directory + 'Testing/Public/images/')
            image_name = data_directory + 'Testing/Public/images/' + image_names[idx-self.cellseg_len[0]]
            mask_name = data_directory + 'Testing/Public/labels/' + image_names[idx-self.cellseg_len[0]][:12] + '_label.tiff'
        elif idx < self.cellseg_len[0] + self.cellseg_len[1] + self.cellseg_len[2]:
            image_names = os.listdir(data_directory + 'Training/images/')
            image_name = data_directory + 'Training/images/' + image_names[idx-self.cellseg_len[0]-self.cellseg_len[1]]
            mask_name = data_directory + 'Training/labels/' + image_names[idx-self.cellseg_len[0]-self.cellseg_len[1]][:10] + '_label.tiff'
        else:
            image_names = os.listdir(data_directory + 'Tuning/images/')
            image_name = data_directory + 'Tuning/images/' + image_names[idx-self.cellseg_len[0]-self.cellseg_len[1]-self.cellseg_len[2]]
            mask_name = data_directory + 'Tuning/labels/' + image_names[idx-self.cellseg_len[0]-self.cellseg_len[1]-self.cellseg_len[2]][:10] + '_label.tiff'
        return image_name, mask_name

    def get_cocahis(self, idx):
        data_directory = os.path.join(self.data_directory, 'CoCaHis/')
        #image_names = os.listdir(data_directory + 'images/')
        image_name = data_directory + 'images/' + "HE_raw_" + str(idx) + ".png"
        if self.cluster != "denbi":
            mask_name = data_directory + 'GT/' + "GT_GT_majority_vote_" + str(idx) + ".png"
        else:
            mask_name = data_directory + 'labels/' + 'GT_GT_majority_vote_' + str(idx) + ".png"
        return image_name, mask_name

    def get_conic(self, idx):
        data_directory = os.path.join(self.data_directory, 'CoNIC/')
        mask_names = os.listdir(data_directory + 'labels_png/')
        image_name = data_directory + 'images_png/' + mask_names[idx][:4] + ".png"
        mask_name = data_directory + 'labels_png/' + mask_names[idx]
        return image_name, mask_name

    def get_crag(self, idx):
        data_directory = os.path.join(self.data_directory, "CRAG/cell_CRAG/")
        if idx < self.crag_len[0]:
            mask_names = os.listdir(data_directory + 'train2017/labels/')
            image_name = data_directory + 'train2017/' + mask_names[idx]
            mask_name = data_directory + 'train2017/labels/' + mask_names[idx]
        else:
            mask_names = os.listdir(data_directory + 'val2017/labels/')
            image_name = data_directory + 'val2017/' + mask_names[idx-self.crag_len[0]]
            mask_name = data_directory + 'val2017/labels/' + mask_names[idx-self.crag_len[0]]
        return image_name, mask_name
    
    def get_cryonuseg(self, idx):
        data_directory = os.path.join(self.data_directory, 'CryoNuSeg/')
        image_names = os.listdir(data_directory + 'tissue images/')
        image_name = data_directory + 'tissue images/' + image_names[idx]
        mask_name = data_directory + 'Annotator 1 (biologist second round of manual marks up)/Annotator 1 (biologist second round of manual marks up)/label masks modify/' + image_names[idx]
        return image_name, mask_name

    def get_glas(self, idx):
        data_directory = os.path.join(self.data_directory, 'GlaS/Warwick_QU_Dataset/')
        if idx < 60:
            image_name = data_directory + 'testA_' + str(idx + 1) + '.bmp'
            mask_name = data_directory + 'testA_' + str(idx + 1) + '_anno.bmp'
        elif idx < 60 + 20:
            image_name = data_directory + 'testB_' + str(idx - 59) + '.bmp'
            mask_name = data_directory + 'testB_' + str (idx - 59) + '_anno.bmp'
        else:
            image_name = data_directory + 'train_' + str(idx - 79) + '.bmp'
            mask_name = data_directory + 'train_' + str(idx - 79) + '_anno.bmp'
        return image_name, mask_name

    def get_icia2018(self, idx):
        data_directory = os.path.join(self.data_directory, 'ICIA2018/ICIAR2018_BACH_Challenge/WSI/')
        image_name = data_directory + 'A' + str(idx + 1).zfill(2) + '.svs'
        mask_name = data_directory + 'A' + str(idx + 1).zfill(2) + '.npy'
        return image_name, mask_name

    def get_janowczyk(self, idx):
        data_directory = os.path.join(self.data_directory, 'Janowczyk/')
        if self.cluster != "denbi":
            names = os.listdir(data_directory)
            image_names = [n for n in names if "original" in n]
            image_names.sort()
            image_name = data_directory + image_names[idx]
            mask_name = data_directory + image_names[idx][:-12] + 'mask.png'
        else:
            image_names = os.listdir(data_directory + "images/")
            image_names.sort()
            image_name = data_directory + "images/" + image_names[idx]
            mask_name = data_directory + "masks/" + image_names[idx][:-12] + 'mask.png'
        return image_name, mask_name
    
    def get_kpi(self, idx):
        if self.cluster != "denbi":
            data_directories = [os.path.join(self.data_directory, 'KPI/', 'Task1_patch_level/data'), os.path.join(self.data_directory, 'KPI/', 'val/Task1_patch_level/data')]
        else:
            data_directories = [os.path.join(self.data_directory, 'KPI/', "KPIs24 Training Data/Task1_patch_level/data"), os.path.join(self.data_directory, 'KPI/', "KPIs24 Validation Data/Task1_patch_level/data")]
        position = 0
        if idx < self.length[0]:
            data_directory = data_directories[0]
            position = self.pointer[0]
        else:
            data_directory = data_directories[1]
            idx -= self.length[0]
            position = self.pointer[1]
        dir_list = sorted(os.listdir(data_directory))
        for (i1, d1) in enumerate(dir_list):
            if idx < self.length[position + i1]:
                position = self.pointer[position + i1]
                subdir_list = sorted(os.listdir(os.path.join(data_directory, d1)))
                for (i2, d2) in enumerate(subdir_list):
                    if idx < self.length[position + i2]:
                        file_list = sorted(os.listdir(os.path.join(data_directory, d1, d2,'img')))
                        image_name = os.path.join(data_directory, d1, d2, 'img', file_list[idx])
                        mask_name = os.path.join(data_directory, d1, d2, 'mask', file_list[idx].replace('img', 'mask'))
                        return image_name, mask_name
                    else:
                        idx -= self.length[position + i2]          
            else:
                idx -= self.length[position + i1]

    def get_kumar(self, idx):
        data_directory = os.path.join(self.data_directory, 'Kumar/')
        if idx < self.kumar_len[0]:
            mask_names = os.listdir(data_directory + 'train/Labels_png/')
            image_name = data_directory + 'train/Images/' + mask_names[idx][:23] + ".tif"
            mask_name = data_directory + 'train/Labels_png/' + mask_names[idx]
        elif idx < self.kumar_len[0] + self.kumar_len[1]:
            mask_names = os.listdir(data_directory + 'test_same/Labels_png/')
            image_name = data_directory + 'test_same/Images/' + mask_names[idx-self.kumar_len[0]][:23] + ".tif"
            mask_name = data_directory + 'test_same/Labels_png/' + mask_names[idx-self.kumar_len[0]]
        else:
            mask_names = os.listdir(data_directory + 'test_diff/Labels_png/')
            image_name = data_directory + 'test_diff/Images/' + mask_names[idx-self.kumar_len[0]-self.kumar_len[1]][:23] + ".tif"
            mask_name = data_directory + 'test_diff/Labels_png/' + mask_names[idx-self.kumar_len[0]-self.kumar_len[1]]
        return image_name, mask_name

    def get_monusac(self, idx):
        data_directory = os.path.join(self.data_directory, 'MoNuSAC/')
        mask_names = os.listdir(data_directory + 'masks/')
        mask_name = data_directory + 'masks/' + mask_names[idx]
        types = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"]
        image_name = data_directory + 'images/' + mask_names[idx]
        for t in types:
            if mask_names[idx].endswith(t + '.png'):
                image_name = data_directory + 'images/' + mask_names[idx][:-len(t)-5]
                if os.path.exists(image_name + '.tif'):
                    image_name += '.tif'
                else:
                    image_name += '.png'
                break       
        return image_name, mask_name

    def get_monuseg(self, idx):
        data_directory = os.path.join(self.data_directory, 'MoNuSeg/MoNuSeg 2018 Training Data/')
        mask_names = os.listdir(data_directory + 'Masks/')
        mask_name = data_directory + 'Masks/' + mask_names[idx]
        image_name = data_directory + 'Tissue Images/' + mask_names[idx][:23] + '.tif'
        return image_name, mask_name

    def get_nuclick(self, idx):
        data_directory = os.path.join(self.data_directory, 'NuClick/')
        if idx < 1213:
            mask_names = os.listdir(data_directory + 'Hemato_Data/Train/masks/')
            image_name = data_directory + 'Hemato_Data/Train/images/' + mask_names[idx][:-9] + ".png"
            mask_name = data_directory + 'Hemato_Data/Train/masks/' + mask_names[idx]
        elif idx < 1213 + 250:
            mask_names = os.listdir(data_directory + 'Hemato_Data/Validation/masks/')
            image_name = data_directory + 'Hemato_Data/Validation/images/' + mask_names[idx-1213][:-9] + ".png"
            mask_name = data_directory + 'Hemato_Data/Validation/masks/' + mask_names[idx-1213]
        elif idx < 1213 + 250 + 462:
            mask_names = os.listdir(data_directory + 'IHC_nuclick/IHC/masks_png/Train/')
            image_name = data_directory + 'IHC_nuclick/IHC/images/Train/' + mask_names[idx-1213-250]
            mask_name = data_directory + 'IHC_nuclick/IHC/masks_png/Train/' + mask_names[idx-1213-250]
        else:
            mask_names = os.listdir(data_directory + 'IHC_nuclick/IHC/masks_png/Validation/')
            image_name = data_directory + 'IHC_nuclick/IHC/images/Validation/' + mask_names[idx-1213-250-462]
            mask_name = data_directory + 'IHC_nuclick/IHC/masks_png/Validation/' + mask_names[idx-1213-250-462]
        return image_name, mask_name

    def get_paip2023(self, idx):
        data_directory = os.path.join(self.data_directory, 'PAIP2023/')
        if idx < 50:
            image_name = data_directory + 'tr_p' + str(idx + 1).zfill(3) + '.png'
            mask_name = data_directory + "non_tumor/" + 'tr_p' + str(idx + 1).zfill(3) + '_nontumor.png'
        elif idx < 50 + 50:
            image_name = data_directory + 'tr_p' + str(idx - 50 + 1).zfill(3) + '.png'
            mask_name = data_directory + "tumor/" + 'tr_p' + str(idx - 50 + 1).zfill(3) + '_tumor.png'
        elif idx < 50 + 50 + 3:
            image_name = data_directory + 'tr_c' + str(idx - 100 + 1).zfill(3) + '.png'
            mask_name = data_directory + "non_tumor/" + 'tr_c' + str(idx -100 + 1).zfill(3) + '_nontumor.png'
        else:
            image_name = data_directory + 'tr_c' + str(idx - 103 + 1).zfill(3) + '.png'
            mask_name = data_directory + "tumor/" + 'tr_c' + str(idx -103 + 1).zfill(3) + '_tumor.png'
        return image_name, mask_name

    def get_pannuke(self, idx):
        data_directory = os.path.join(self.data_directory, 'PanNuke/')
        image_name = data_directory + 'images_png/' + str(idx).zfill(4) + ".png"
        mask_name = data_directory + 'masks_png/' + str(idx).zfill(4) + ".png"
        return image_name, mask_name

    def get_segpath(self, idx):
        data_directory = os.path.join(self.data_directory, 'SegPath/')
        if idx < 10647:
            image_and_mask_names = os.listdir(data_directory + 'endothelial_cells/ERG_Endothelium/')
            image_names = [x for x in image_and_mask_names if "HE" in x]
            image_name = data_directory + 'endothelial_cells/ERG_Endothelium/' + image_names[idx]
            mask_name = data_directory + 'endothelial_cells/ERG_Endothelium/' + image_names[idx][:-6] + 'mask.png' 
        elif idx < 10647 + 26509:
            image_and_mask_names = os.listdir(data_directory + 'epithelial_cells/panCK_Epithelium/')
            image_names = [x for x in image_and_mask_names if "HE" in x]
            image_name = data_directory + 'epithelial_cells/panCK_Epithelium/' + image_names[idx-10647]
            mask_name = data_directory + 'epithelial_cells/panCK_Epithelium/' + image_names[idx-10647][:-6] + 'mask.png'
        elif idx < 10647 + 26509 + 24805:
            image_and_mask_names = os.listdir(data_directory + 'leukocytes/CD45RB_Leukocyte/')
            image_names = [x for x in image_and_mask_names if "HE" in x]
            image_name = data_directory + 'leukocytes/CD45RB_Leukocyte/' + image_names[idx-10647-26509]
            mask_name = data_directory + 'leukocytes/CD45RB_Leukocyte/' + image_names[idx-10647-26509][:-6] + 'mask.png'
        elif idx < 10647 + 26509 + 24805 + 12273:
            image_and_mask_names = os.listdir(data_directory + 'lymphocytes/CD3CD20_Lymphocyte/')
            image_names = [x for x in image_and_mask_names if "HE" in x]
            image_name = data_directory + 'lymphocytes/CD3CD20_Lymphocyte/' + image_names[idx-10647-26509-24805]
            mask_name = data_directory + 'lymphocytes/CD3CD20_Lymphocyte/' + image_names[idx-10647-26509-24805][:-6] + 'mask.png'
        elif idx < 10647 + 26509 + 24805 + 12273 + 14135:
            image_and_mask_names = os.listdir(data_directory + 'myeloid_cells/MNDA_MyeloidCell/')
            image_names = [x for x in image_and_mask_names if "HE" in x]
            image_name = data_directory + 'myeloid_cells/MNDA_MyeloidCell/' + image_names[idx-10647-26509-24805-12273]
            mask_name = data_directory + 'myeloid_cells/MNDA_MyeloidCell/' + image_names[idx-10647-26509-24805-12273][:-6] + 'mask.png'
        elif idx < 10647 + 26509 + 24805 + 12273 + 14135 + 13231:
            image_and_mask_names = os.listdir(data_directory + 'plasma_cells/MIST1_PlasmaCell/')
            image_names = [x for x in image_and_mask_names if "HE" in x]
            image_name = data_directory + 'plasma_cells/MIST1_PlasmaCell/' + image_names[idx-10647-26509-24805-12273-14135]
            mask_name = data_directory + 'plasma_cells/MIST1_PlasmaCell/' + image_names[idx-10647-26509-24805-12273-14135][:-6] + 'mask.png'
        elif idx < 10647 + 26509 + 24805 + 12273 + 14135 + 13231 + 25909:
            image_and_mask_names = os.listdir(data_directory + 'red_blood_cells/CD235a_RBC/')
            image_names = [x for x in image_and_mask_names if "HE" in x]
            image_name = data_directory + 'red_blood_cells/CD235a_RBC/' + image_names[idx-10647-26509-24805-12273-14135-13231]
            mask_name = data_directory + 'red_blood_cells/CD235a_RBC/' + image_names[idx-10647-26509-24805-12273-14135-13231][:-6] + 'mask.png'
        else:
            image_and_mask_names = os.listdir(data_directory + 'smooth_muscle_cells/aSMA_SmoothMuscle/')
            image_names = [x for x in image_and_mask_names if "HE" in x]
            image_name = data_directory + 'smooth_muscle_cells/aSMA_SmoothMuscle/' + image_names[idx-10647-26509-24805-12273-14135-13231-25909]
            mask_name = data_directory + 'smooth_muscle_cells/aSMA_SmoothMuscle/' + image_names[idx-10647-26509-24805-12273-14135-13231-25909][:-6] + 'mask.png'
        return image_name, mask_name

    def get_segpc(self, idx):
        data_directory = os.path.join(self.data_directory, 'SegPC/TCIA_SegPC_dataset/')
        if idx < 298:
            mask_names = os.listdir(data_directory + 'train/masks_png/')
            image_name = data_directory + 'train/x/' + mask_names[idx][:-4] + ".bmp"
            mask_name = data_directory + 'train/masks_png/' + mask_names[idx]
        else:
            mask_names = os.listdir(data_directory + 'validation/masks_png/')
            image_name = data_directory + 'validation/x/' + mask_names[idx-298][:-4] + ".bmp"
            mask_name = data_directory + 'validation/masks_png/' + mask_names[idx-298]
        return image_name, mask_name

    def get_tiger(self, idx):
        data_directory = os.path.join(self.data_directory, 'TIGER/wsirois/roi-level-annotations/tissue-cells/')
        image_names = os.listdir(data_directory + 'images/')
        image_name = data_directory + 'images/' + image_names[idx]
        mask_name = data_directory + 'masks/' + image_names[idx]
        return image_name, mask_name

    def get_tnbc(self, idx):
        if self.cluster == "denbi":
            data_directory = os.path.join(self.data_directory, 'TNBC/TNBC_NucleiSegmentation/')
        else: 
            data_directory = os.path.join(self.data_directory, 'TNBC/TNBC_dataset/')
        bucket = 1
        idx = idx + 1
        if idx > 7:
            bucket += 1
            idx -= 7
            if idx > 3:
                bucket += 1
                idx -= 3
                if idx > 5:
                    bucket += 1
                    idx -= 5
                    if idx > 8:
                        bucket += 1
                        idx -= 8
                        if idx > 4:
                            bucket += 1
                            idx -= 4
                            if idx > 3:
                                bucket += 1
                                idx -= 3
                                if idx > 3:
                                    bucket += 1
                                    idx -= 3
                                    if idx > 4:
                                        bucket += 1
                                        idx -= 4
                                        if idx > 6:
                                            bucket += 1
                                            idx -= 6
                                            if idx > 4:
                                                bucket += 1
                                                idx -= 4
        image_name = data_directory + 'Slide_' + str(bucket).zfill(2) + '/' +  str(bucket).zfill(2) + '_' + str(idx) + '.png'
        mask_name = data_directory + 'GT_' + str(bucket).zfill(2) + '/' +  str(bucket).zfill(2) + '_' + str(idx) + '.png'
        return image_name, mask_name