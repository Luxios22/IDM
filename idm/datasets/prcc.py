from __future__ import print_function, absolute_import
import os
import glob
import random
import re
from ..utils.data import BaseImageDataset


class PRCC(BaseImageDataset):
    """
    PRCC
    Reference:
    Qize Yang et al. Person Re-identification by Contour Sketch under Moderate Clothing Change. IEEE Transactions on Pattern Analysis and Machine Intelligence.
    URL: https://www.isee-ai.cn/~yangqize/clothing.html

    Dataset statistics:
    # identities: 221
    # images: 33698
    # from each camera select a image randomly as the query image for each pid
    """
    dataset_dir = 'prcc/rgb'

    def __init__(self, root, verbose=True, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.train_dir = os.path.join(self.dataset_dir, 'train')
        self.query_dir = os.path.join(self.dataset_dir, 'test')
        self.gallery_dir = os.path.join(self.dataset_dir, 'test')
        self.cam2label = {'A': 0, 'B': 1, 'C': 2}

        self._check_before_run()

        self.train = self._process_dir(self.train_dir, relabel=True, is_train=True, is_query=False)
        self.query = self._process_dir(self.query_dir, relabel=False, is_train=False, is_query=True)
        self.gallery = self._process_dir(self.gallery_dir, relabel=False, is_train=False, is_query=False)

        if verbose:
            print("=> PRCC Loaded")
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

    def _process_dir(self, dir_path, relabel=False, is_train=False, is_query=False):
        dataset = []
        dirs = os.listdir(dir_path)
        
        if is_train:
            pattern = re.compile(r'([A-C])?_?cropped_rgb(\d+)')
            assert len(dirs) == 150 
            pid2label = {int(pid): label for label, pid in enumerate(dirs)}
            for pid in dirs:
                img_paths = glob.glob(os.path.join(dir_path, pid, '*.jpg'))
                if relabel: pid = pid2label[int(pid)]
                for img_path in img_paths:
                    cam, _ = pattern.search(img_path).groups()
                    camid = self.cam2label[cam]
                    assert 0 <= camid <= 2
                    dataset.append((img_path, pid, camid))
        else:
            assert len(dirs) == 3
            for cam in dirs:
                camid = self.cam2label[cam]
                pids = os.listdir(os.path.join(dir_path, cam))
                assert 0 <= camid <= 2
                assert len(pids) == 71
                pid2label = {pid: label for label, pid in enumerate(pids)}
                for pid in pids:
                    img_paths = glob.glob(os.path.join(dir_path, cam, pid, '*.jpg'))
                    if relabel: pid = pid2label[pid]
                    if is_query: dataset.append((random.choice(img_paths), int(pid), camid))
                    else:
                        for img_path in img_paths:
                            dataset.append((img_path, int(pid), camid))

        return dataset
