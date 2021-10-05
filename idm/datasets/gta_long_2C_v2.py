# encoding: utf-8
"""
ROSE Lab's GTA ReID Dataset, v3 alpha. This is the train_long-full version, meaning that there is no separation between
training set, query set and gallery set. The training set consists of all annotated images available. This is to
facillitate cross domain experiments.
"""
from __future__ import print_function, absolute_import
import os
from ..utils.data import BaseImageDataset


class GTA_Long_2C_V2(BaseImageDataset):
    """
    GTA Long-Term

    Dataset statistics:
    # identities: 690 unique PID
    # appearances: 1972
    # images 11,358,928
    
    """
    dataset_dir = 'GTA_Long'

    def __init__(self, root, verbose = True, relabel = False, **kwargs):
        
        super(GTA_Long_2C_V2, self).__init__()

        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_list = os.path.join(self.dataset_dir, 'train_long_2C_v2.txt')
        self.query_list = os.path.join(self.dataset_dir, f'query_long_2C_v2.txt')
        self.gallery_list = os.path.join(self.dataset_dir, f'gallery_long_2C_v2.txt')
        print(f'The dataset directory is: {self.dataset_dir}')
        
        pid_files = ['train_long_2C_v2.txt',f'gallery_long_2C_v2.txt']

        for fil in pid_files:
            print(os.path.exists(os.path.join(self.dataset_dir, fil)))
            if not os.path.exists(os.path.join(self.dataset_dir, fil)):
                print(f'Labels for {fil} have not been generated.')
                print(f'Labels will be generated into dir {self.dataset_dir}')
                self.generate_label(fil)
        # if cfg.FILTERS.QUERY_ANGLE:
        #     print(f"You are currently doing query on {self.qname}")

        self.check_before_run()
        self.cam_dict = {
            'beach' : 1,
            'carpark' : 2,
            'construction' : 3,
            'countryside' : 4,
            'fbi' : 5,
            'greenery' : 6,
            'indoorcarpark' : 7,
            'pier' : 8,
            'subway' : 9,
            'urbanstreet' : 10,
        }

        train = self._process_list(self.train_list, relabel=True)
        gallery = self._process_list(self.gallery_list, relabel=False)
        query = self._process_list(self.query_list, relabel=False)

        if verbose:
            print(f"=> {self.dataset_dir} loaded. If you did not intend for the training set to consist of the full images, ABORT NOW!")
            self.print_dataset_statistics(train, query, gallery)
            

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    # def generate_label(self,fil):
    #     #This is where to add filters
    #     save_dir = self.dataset_dir
    #     img_format = '.jpg'
    #     loc_filter = self.filters[0]
    #     f = open(os.path.join(save_dir,fil),'w')
    #     if fil == f'gallery_1-3.txt':
    #         q_file =  open(os.path.join(save_dir,f'query_0.txt'),'w')

    #     if not os.path.exists(self.dataset_dir):
    #         print('Dataset not downloaded')
        
    #     for root, subdir, files in os.walk(self.dataset_dir):
    #         ped_list = []
    #         ped_list_s = []
    #         if self.gta_uniform:
    #             if self.problematic(root):
    #                 continue
    #         for _file in files:
    #             if _file.endswith('.txt') or _file.endswith('.xml'):
    #                 continue
    #             if _file.endswith(img_format):
    #                 _filestr = _file.replace(img_format,"") #removes the .jpg portion of the filename
    #                 _file_split = _filestr.split('_')
    #                 ped = self.create_pedlist(root, _file, fil, _file_split)
    #                 if ped == 0:
    #                     continue
    #                 else:
    #                     ped_list.append(ped)
    #         ped_list = np.array(ped_list)
    #         if ped_list.size == 0:
    #             continue
            
    #         #ped_list = np.random.choice(ped_list,100, replace = False)
    #         # ped_list = np.random.choice(ped_list,int(0.125*len(ped_list)), replace = False) #chooses a random sample of 50 images for the dataset, with replacement (i.e. there can be duplicates.)
    #         # q_list = np.random.choice(ped_list,int(0.2*len(ped_list)))

    #         for q in ped_list:
    #             q_str = q.replace(img_format,"")
    #             q = q.replace(img_format,"")
    #             q_split = q_str.split('_')
    #             if self.filters[-1] != []:
    #                 min_yaw = str(min(self.filters[-1]))
    #             else:
    #                 min_yaw = '1'
    #             # if fil == f'gallery_{self.qname}.txt':
    #             if q_split[-1] == min_yaw: #This should only execute once for each set of conditions.

    #                 if self.q_angles:
    #                     for ang in self.q_angles:
    #                         q_split[-1] = str(ang)
    #                         q = '_'.join(q_split)+'.jpg'
    #                         if fil == f'gallery_1-3.txt':
    #                             print(q, file = q_file)

    #             # while int(q_split[-1]) == int(q.split('_')[-1]):
    #             #     q_split[-1] = str(np.random.randint(1, high=9))
    #             # if (q.split('_')[-3]) == 'person':
    #             #     q_split[-3] = 'surveillance'
    #             # else:
    #             #     q_split[-3] = 'person'



    #         for ped in ped_list:
    #             print(ped, file = f)
            


    #     f.close()
    #     if fil == f'gallery_1-3.txt':
    #         q_file.close()
    #     # if self.gta_uniform:
    #     #     self.remove_duplicates(fil)
    #     #     self.remove_duplicates(f'query_{self.qname}.txt')


    # def create_pedlist(self, root, _file, _filestr, _file_split):
    #     filters = self.filters
    #     for idx, fil in enumerate(filters):
    #         if fil:
    #             if _file_split[idx] not in fil:
    #                 return 0
    #         ped = os.path.join(''.join(root.split('/')[-2:-1]),_file_split[0], _file)
    #     return ped
        

    def check_before_run(self):
        '''Check that all files are available before going deeper'''
        checklist = [self.dataset_dir, self.train_list, self.query_list, self.gallery_list]
        for item in checklist:
            if not os.path.exists(item):
                raise RuntimeError(f"'{item}' is not available.")

    def _process_list(self, list_path, relabel=False, camid=-1):
        with open(list_path, 'r') as f: 
            img_paths = [l.strip() for l in f.readlines()]
        dataset = [] 
        if relabel: 
            pid_container = set()
            for img_path in img_paths: 
                #example img_path: /home/GTA/GTA_Long/1/beach/beach_CLEAR_13_walk_surveillance_1_6.jpg
                pid = int(img_path.split('_')[-2]) 
                #pid = 1 for the above example
                pid_container.add(pid)
            pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path in img_paths:
            pid = int(img_path.split('_')[-2])
            camid = self.cam_dict[img_path.split('/')[-2]]
            assert 1 <= camid <= 10
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset

    # def remove_duplicates(self,fil):
    #     pid_dict = {}
    #     with open(os.path.join(self.dataset_dir, fil),'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             line = line.strip('\n')
    #             line_us_split = line.split('_')
    #             line_slash_split = line.split('/')
    #             if line_us_split[-2] not in pid_dict:
    #                 pid_dict[line_us_split[-2]] = set()
    #             if not self.problematic(line_slash_split[0]):
    #                 pid_dict[line_us_split[-2]].add(int(line_slash_split[0]))
    #     print(pid_dict)

    #     with open(os.path.join(self.dataset_dir, fil),'w') as f:
    #         for line in lines:
    #             line = line.strip('\n')
    #             line_us_split = line.split('_')
    #             line_slash_split = line.split('/')
    #             try:
    #                 long_pid = str(min(pid_dict[line_us_split[-2]]))
    #             except ValueError:
    #                 print(f'{pid_dict[line_us_split[-2]]=}')
    #                 continue
    #             if line_slash_split[0] == long_pid:
    #                 print(line,file = f)
    
    # def problematic(self,longid):
    #     problem_files = [1006,1056,1068,1078,1105,1111,1153,1171,1180,1192,1204,1209,
    #     1215,1218,1234,1239,1253,1286,1296,1324,1326,1331,1356,1373,1397,1405,1437,1479,
    #     1509,1513,1518,1524,1531,1547,1576,1598,1605,1614,1620,1631,1640,1645,1678,1707,
    #     1731,1745,1810,1830,1890,1895,1901,1906,1938,1949,1961,940,938,937]
    #     problem_files_str = [str(f) for f in problem_files]
        
    #     return longid in problem_files_str




    
