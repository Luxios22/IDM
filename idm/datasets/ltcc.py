from __future__ import print_function, absolute_import
import os
import glob
import re
from ..utils.data import BaseImageDataset


class LTCC(BaseImageDataset):
    """
    LTCC
    Reference:
    Qian et al. Long-Term Cloth-Changing Person Re-identification. ICCV 2015.
    URL: https://naiq.github.io/LTCC_Perosn_ReID.html

    Dataset statistics:
    # identities: 152 (91 for cloth-change, 61 for cloth-consistent)
    # images:  9576 (train) + 493 (query) + 7050 (gallery)
    """
    dataset_dir = 'LTCC_ReID'

    def __init__(self, root, verbose=True, **kwargs):
        super(LTCC, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.query_dir = os.path.join(self.dataset_dir, 'query')
        self.gallery_dir = os.path.join(self.dataset_dir, 'test')
        self.info_dir = os.path.join(self.dataset_dir, 'info')

        self._check_before_run()

        with open(os.path.join(self.info_dir, 'cloth-change_id_test.txt'), 'r') as f1, \
            open(os.path.join(self.info_dir, 'cloth-change_id_train.txt'), 'r') as f2, \
                open(os.path.join(self.info_dir, 'cloth-unchange_id_test.txt'), 'r') as f3, \
                    open(os.path.join(self.info_dir, 'cloth-unchange_id_train.txt'), 'r') as f4:
                    self.train_long_list = [line.strip('\n') for line in f2.readlines()]
                    self.train_short_list = [line.strip('\n') for line in f4.readlines()]
                    self.test_long_list = [line.strip('\n') for line in f1.readlines()]
                    self.test_short_list = [line.strip('\n') for line in f3.readlines()]

        self.train = self._process_dir(self.train_dir, relabel=True, mix=False, train=True, long=True)
        self.query = self._process_dir(self.query_dir, relabel=False, mix=False, train=False, long=True)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False, mix=False, train=False, long=True)

        if verbose:
            print("=> LTCC Loaded")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not os.path.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not os.path.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not os.path.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not os.path.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, mix=False, train=False, long=False):
        # /home/GTA/LTCC_ReID/train/140_2_c5_002106.png
        img_paths = glob.glob(os.path.join(dir_path, '*.png'))
        pattern = re.compile(r'([\d]+)_([\d]+)_c(\d)')

        pid_container = set()
        img_list = []
        for img_path in img_paths:
            pid, _, camid = pattern.search(img_path).groups()
            valid = False
            if mix: valid = True
            elif train and long: 
                if pid in self.train_long_list: valid = True
            elif train and not long:
                if pid in self.train_short_list: valid = True
            elif not train and long:
                if pid in self.test_long_list: valid = True
            elif not train and not long:
                if pid in self.test_short_list: valid = True            
            if valid: 
                pid_container.add(int(pid))
                img_list.append([img_path, int(pid), int(camid)])
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img in img_list:
            img_path, pid, camid = img
            assert 0 <= pid <= 151
            assert 1 <= camid <= 12
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset
