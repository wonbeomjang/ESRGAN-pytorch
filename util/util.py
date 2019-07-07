import os
import shutil
import urllib
import zipfile
import tarfile

from tqdm import tqdm


class LambdaLR:
    def __init__(self, n_epoch, offset, total_batch_size, decay_batch_size):
        self.n_epoch = n_epoch
        self.offset = offset
        self.total_batch_size = total_batch_size
        self.decay_batch_size = decay_batch_size

    def step(self, epoch):
        factor = pow(0.5, int(((self.offset + epoch) * self.total_batch_size) / self.decay_batch_size))
        return factor


def unzip_zip_file(zip_path, data_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(zip_path)


def unzip_tar_file(zip_path, data_path):
    tar_ref = tarfile.open(zip_path, "r:")
    tar_ref.extractall(data_path)
    tar_ref.close()
    os.remove(zip_path)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print("[!] download data file")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def reformat_file(dataset_path):
    dir = os.listdir(dataset_path)

    for data_scale_dir in dir:
        data_scale_dir = os.path.join(dataset_path, data_scale_dir)
        data_file = os.listdir(data_scale_dir)

        print(f"[!] Move file in {data_scale_dir}")
        for file in tqdm(data_file):
            src_path = os.path.join(data_scale_dir, file)
            des_path = os.path.join(dataset_path, file)
            shutil.move(src_path, des_path)
        os.rmdir(data_scale_dir)
