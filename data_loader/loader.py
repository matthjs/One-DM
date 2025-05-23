import random
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import pickle
from torchvision import transforms
import lmdb
from PIL import Image
import torchvision
import cv2
from einops import rearrange, repeat
import time
import torch.nn.functional as F

text_path = {'train': 'data/IAM64_train.txt',  # this is hardcoded for some reason
              'test': 'data/IAM64_test.txt'}
# text_path = {'train': 'data/dss/dss_train.txt',  # this is hardcoded for some reason
#             'test': 'data/dss/dss_test.txt'}

generate_type = {'iv_s': ['train', 'data/in_vocab.subset.tro.37'],
                'iv_u': ['test', 'data/in_vocab.subset.tro.37'],
                'oov_s': ['train', 'data/oov.common_words'],
                'oov_u': ['test', 'data/oov.common_words']}

# generate_type = {'iv_s': ['train', 'data/dss/in_vocab.subset.tro.37'],
#                  'iv_u': ['test', 'data/dss/in_vocab.subset.tro.37'],
#                  'oov_s': ['train', 'data/dss/oov.common_words'],
#                 'oov_u': ['test', 'data/dss/oov.common_words']}

# define the letters and the width of style image
letters = '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%' \
          # + 'אבגדהוזחטיכךלמםנןסעפףצץקרשת'  # ADDED
style_len = 352

"""prepare the IAM dataset for training"""


class IAMDataset(Dataset):
    def __init__(self, image_path, style_path, laplace_path, type, content_type='unifont', max_len=9):
        """
        Dataset for loading IAM handwriting data with associated style and Laplacian references.

        :param image_path: Path to the root image directory.
        :param style_path: Path to the root style image directory.
        :param laplace_path: Path to the root Laplacian image directory.
        :param type: Either 'train' or 'test', used to determine data split.
        :param content_type: Type of font used for generating content images (e.g. 'unifont').
        :param max_len: Maximum allowed sequence length for the text.
        """
        self.max_len = max_len
        self.style_len = style_len
        self.data_dict = self.load_data(text_path[type])
        self.image_path = os.path.join(image_path, type)
        self.style_path = os.path.join(style_path, type)
        self.laplace_path = os.path.join(laplace_path, type)

        self.letters = letters
        self.tokens = {"PAD_TOKEN": len(self.letters)}
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.indices = list(self.data_dict.keys())
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # self.content_transform = torchvision.transforms.Resize([64, 32], interpolation=Image.NEAREST)
        self.con_symbols = self.get_symbols(content_type)
        self.laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float
                                    ).to(torch.float32).view(1, 1, 3, 3).contiguous()
        self.prepare_laplacian_images()

    def prepare_laplacian_images(self):
        """
        Precomputes Laplacian-filtered versions of all style images if not already present.
        Saves them to `self.laplace_path`.
        """
        print("[INFO] Checking for missing Laplacian images...")
        for wr_id in os.listdir(self.style_path):
            wr_style_dir = os.path.join(self.style_path, wr_id)
            wr_laplace_dir = os.path.join(self.laplace_path, wr_id)

            if not os.path.exists(wr_laplace_dir):
                os.makedirs(wr_laplace_dir)

            for fname in os.listdir(wr_style_dir):
                style_img_path = os.path.join(wr_style_dir, fname)
                laplace_img_path = os.path.join(wr_laplace_dir, fname)

                if os.path.exists(laplace_img_path):
                    continue  # Already exists

                img = cv2.imread(style_img_path, flags=0)  # grayscale
                img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float() / 255.0  # shape [1,1,H,W]
                lap = F.conv2d(img_tensor, self.laplace, padding=1)
                lap = lap.squeeze().numpy()
                # lap = 1.0 - lap  # invert as done in the loader
                lap = np.clip(lap * 255.0, 0, 255).astype(np.uint8)

                cv2.imwrite(laplace_img_path, lap)

        print("[INFO] Laplacian precomputation done.")

    def load_data(self, data_path):
        """
        Loads label data from file.

        :param data_path: Path to the label file.
        :return: Dictionary containing image metadata (label and writer ID).
        """
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split(' ') for i in train_data]
            full_dict = {}
            idx = 0
            for i in train_data:
                s_id = i[0].split(',')[0]
                image = i[0].split(',')[1] + '.png'
                transcription = i[1]
                if len(transcription) > self.max_len:
                    continue
                full_dict[idx] = {'image': image, 's_id': s_id, 'label': transcription}
                idx += 1
        return full_dict

    def get_style_ref(self, wr_id):
        """
        Randomly selects two style reference images and their Laplacians for a given writer.

        :param wr_id: Writer ID.
        :return: Tuple of (style_images, laplace_images) as numpy arrays of shape [2, H, W].
        """
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2)  # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
                          for index in style_index]

        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])

        '''style images'''
        style_images = [style_image / 255.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        laplace_images = [laplace_image / 255.0 for laplace_image in laplace_images]
        new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        return new_style_images, new_laplace_images

    def get_symbols(self, input_type):
        """
        Loads and processes content font symbols.

        :param input_type: Font type (e.g., 'unifont').
        :return: Tensor of shape [num_symbols+1, H, W] where the last is the PAD symbol.
        """
        with open(f"data/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0]))  # blank image as PAD_TOKEN
        contents = torch.stack(contents)
        return contents

    def __len__(self):
        return len(self.indices)

    ### Borrowed from GANwriting ###
    def label_padding(self, labels, max_len):
        """
        Pads a sequence of character labels to max_len with PAD_TOKEN.

        :param labels: Input string label.
        :param max_len: Target sequence length.
        :return: List of indices padded to max_len.
        """
        ll = [self.letter2index[i] for i in labels]
        num = max_len - len(ll)
        if not num == 0:
            ll.extend([self.tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
        return ll

    def __getitem__(self, idx):
        """
        Retrieves and processes a single example.

        :param idx: Index of the sample.
        :return: Dictionary containing image, content label, style reference, Laplacian, etc.
        """
        image_name = self.data_dict[self.indices[idx]]['image']
        label = self.data_dict[self.indices[idx]]['label']
        wr_id = self.data_dict[self.indices[idx]]['s_id']
        transcr = label
        img_path = os.path.join(self.image_path, wr_id, image_name)
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)

        style_ref, laplace_ref = self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32)  # [2, h , w] achor and positive
        laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32)  # [2, h , w] achor and positive

        return {'img': image,
                'content': label,
                'style': style_ref,
                "laplace": laplace_ref,
                'wid': int(wr_id),
                'transcr': transcr,
                'image_name': image_name}

    def collate_fn_(self, batch):
        """
        Custom collate function for DataLoader.

        :param batch: List of samples from __getitem__.
        :return: Batched dictionary with padded tensors.
        """
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]

        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len

        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)],
                          dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16, 16], dtype=torch.float32)

        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width],
                               dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width],
                                  dtype=torch.float32)
        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)

        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)
            try:
                content = [self.letter2index[i] for i in item['content']]
                content = self.con_symbols[content]
                content_ref[idx, :len(content)] = content
            except:
                print('content', item['content'])

            target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx]])

            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
            except:
                print('style', item['style'].shape)

        wid = torch.tensor([item['wid'] for item in batch])
        content_ref = 1.0 - content_ref  # invert the image
        return {'img': imgs, 'style': style_ref, 'content': content_ref, 'wid': wid, 'laplace': laplace_ref,
                'target': target, 'target_lengths': target_lengths, 'image_name': image_name}


"""random sampling of style images during inference"""


class Random_StyleIAMDataset(IAMDataset):
    def __init__(self, style_path, lapalce_path, ref_num) -> None:
        """
        Randomly selects two style reference images and their Laplacians from the entire dataset.

        :param wr_id: (Unused) Writer ID — ignored for random sampling.
        :return: Tuple of (style_images, laplace_images) as numpy arrays of shape [2, H, W].
        """
        self.style_path = style_path
        self.laplace_path = lapalce_path
        self.author_id = os.listdir(os.path.join(self.style_path))
        self.style_len = style_len
        self.ref_num = ref_num

    def __len__(self):
        return self.ref_num

    def get_style_ref(self, wr_id):  # Choose the style image whose length exceeds 32 pixels
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        random.shuffle(style_list)
        for index in range(len(style_list)):
            style_ref = style_list[index]

            style_image = cv2.imread(os.path.join(self.style_path, wr_id, style_ref), flags=0)
            laplace_image = cv2.imread(os.path.join(self.laplace_path, wr_id, style_ref), flags=0)
            if style_image.shape[1] > 128:
                break
            else:
                continue
        style_image = style_image / 255.0
        laplace_image = laplace_image / 255.0
        return style_image, laplace_image

    def __getitem__(self, _):
        batch = []
        for idx in self.author_id:
            style_ref, laplace_ref = self.get_style_ref(idx)
            style_ref = torch.from_numpy(style_ref).unsqueeze(0)
            style_ref = style_ref.to(torch.float32)
            laplace_ref = torch.from_numpy(laplace_ref).unsqueeze(0)
            laplace_ref = laplace_ref.to(torch.float32)
            wid = idx
            batch.append({'style': style_ref, 'laplace': laplace_ref, 'wid': wid})

        s_width = [item['style'].shape[2] for item in batch]
        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len
        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width],
                               dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width],
                                  dtype=torch.float32)
        wid_list = []
        for idx, item in enumerate(batch):
            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
                wid_list.append(item['wid'])
            except:
                print('style', item['style'].shape)

        return {'style': style_ref, 'laplace': laplace_ref, 'wid': wid_list}


"""prepare the content image during inference"""


class ContentData(IAMDataset):
    def __init__(self, content_type='unifont') -> None:
        """
        Dataset for generating content tensors from font symbols.

        :param content_type: Font type to load (e.g., 'unifont').
        :param max_len: Maximum number of characters in generated samples.
        """
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.con_symbols = self.get_symbols(content_type)

    def get_content(self, label):
        word_arch = [self.letter2index[i] for i in label]
        content_ref = self.con_symbols[word_arch]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)
