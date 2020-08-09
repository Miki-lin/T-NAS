import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import random
import xml.etree.ElementTree as ET
import copy
import time
from torch.utils.data import DataLoader


classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor']


class VOC(Dataset):
    """
    put mini-imagenet files as :
    root :
        |- images/*.jpg includes all imgeas
        |- train.csv
        |- test.csv
        |- val.csv
    NOTICE: meta-learning is different from general supervised learning, especially the concept of batch and set.
    batch: contains several sets
    sets: conains n_way * k_shot for meta-train set, n_way * n_query for meta-test set.
    """

    def __init__(self, root, mode, batch_size, n_way, k_shot, k_query, resize, anchor, split=1, startidx=0):
        """
        :param root: root path of mini-imagenet
        :param mode: train, val or test
        :param batch_size: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.root = root
        self.mode = mode
        self.anchor = anchor
        self.batch_size = batch_size  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.resize = resize  # resize to
        self.split = split
        self.startidx = startidx  # index label not from 0, but from startidx
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batch_size, n_way, k_shot, k_query, resize))

        if mode == 'train':
            self.transform = transforms.Compose([
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])
        else:
            self.transform = transforms.Compose([
                                                 transforms.Resize((self.resize, self.resize)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                                 ])

        self.voc_path = os.path.join(self.root, 'JPEGImages/')
        xml_path = os.path.join(self.root, 'Annotations/')
        imageset_path = os.path.join(self.root, 'ImageSets/Main/')
        imageset_file = os.path.join(imageset_path, mode + '.txt')
        # classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        #            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        #            'tvmonitor']
        self.dictLabels, self.dictImages = self.load_voc(classes, xml_path, imageset_file)
        self.voc_img2label = {}
        for voc_cls in classes:
            self.voc_img2label[voc_cls] = classes.index(voc_cls)
        self.voc_cls_num = len(self.dictLabels)

        ''' split train data '''
        if self.split != 1:
            data = list(zip(*self.data))
            data_list = [x for data_tuple in data for x in data_tuple]
            label_list = list(range(self.cls_num)) * len(data)
            self.data_target = list(zip(data_list, label_list))
            data_split = self.data_target[int(self.split[0] * len(data_list)):int(self.split[1] * len(data_list))]
            data_dict = {}
            for v in data_split:
                d = v[0]
                lbl = v[1]
                if lbl in data_dict.keys():
                    data_dict[lbl].append(d)
                else:
                    data_dict[lbl] = [d]
            self.data = []
            for i in range(self.cls_num):
                self.data.append(data_dict[i])  # list of list

        self.create_batch(self.batch_size)
        print(self)

    def load_voc(self, classes, xml_path, imageset_file):
        dictLabels = {}
        dictImages = {}
        image_ids = open(imageset_file).read().strip().split()
        for image_id in image_ids:
            xml_file = open(xml_path + '%s.xml' % image_id, encoding='utf-8')
            tree = ET.parse(xml_file)
            root = tree.getroot()
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)
            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                cls = obj.find('name').text
                if cls not in classes or int(difficult) == 1:
                    continue
                cls_id = classes.index(cls)
                xmlbox = obj.find('bndbox')
                b = [int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text),
                     int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text), int(cls_id)]
                c = [float((b[0] + b[2]) / 2.0) / width, float((b[1] + b[3]) / 2.0) / height,
                     float((b[2] - b[0]) / width), float((b[3] - b[1]) / height), int(cls_id)]
                filename = str(image_id + '.jpg')
                label = int(cls_id)

                if label in dictLabels.keys():
                    dictLabels[label].append(filename)
                else:
                    dictLabels[label] = [filename]

                if filename in dictImages.keys():
                    dictImages[filename].append(c)
                else:
                    dictImages[filename] = [c]
        return dictLabels, dictImages

    def create_batch(self, batch_size):
        """
        create batch for meta-learning.
        episode here means batch, and it means how many sets we want to retain.
        :param episodes: batch size
        :return:
        """

        self.support_x_batch = []  # support set batch
        self.query_x_batch = []  # query set batch
        self.support_y_batch = []
        self.query_y_batch = []

        for b in range(batch_size):  # for each batch
            voc_dictLabels = copy.deepcopy(self.dictLabels)
            voc_dictImages = copy.deepcopy(self.dictImages)
            # 1.select n_way classes randomly, choose 5-way from 64 classes, no duplicate
            voc_select_cls = np.random.choice(self.voc_cls_num, self.n_way, replace=False)
            np.random.shuffle(voc_select_cls)
            support_x = []  # list of list
            query_x = []       # list of list

            for voc_cls in voc_select_cls:
                voc_selected_img = np.random.choice(len(voc_dictLabels[voc_cls]), self.k_shot + self.k_query,
                                                    replace=False)
                np.random.shuffle(voc_selected_img)
                voc_indexDtrain = np.array(voc_selected_img[:self.k_shot])
                voc_indexDtest = np.array(voc_selected_img[self.k_shot:])
                # print("batch_size:", b, ",", voc_cls, ":", len(voc_dictLabels[voc_cls]))

                for support_image_id in np.array(voc_dictLabels[voc_cls])[voc_indexDtrain]:
                    while support_image_id in support_x:
                        voc_indexDtrain = np.random.choice(len(voc_dictLabels[voc_cls]), 1, replace=False)
                        support_image_id = np.array(voc_dictLabels[voc_cls])[voc_indexDtrain][0]
                    # support_x.append(np.array(voc_dictLabels[voc_cls])[voc_indexDtrain].tolist())
                    support_x.extend(np.array(voc_dictLabels[voc_cls])[voc_indexDtrain].tolist())

                for query_image_id in np.array(voc_dictLabels[voc_cls])[voc_indexDtest]:
                    while query_image_id in query_x:
                        voc_indexDtest = np.random.choice(len(voc_dictLabels[voc_cls]), 1, replace=False)
                        query_image_id = np.array(voc_dictLabels[voc_cls])[voc_indexDtest][0]
                    # query_x.append(np.array(voc_dictLabels[voc_cls])[voc_indexDtest].tolist())
                    query_x.extend(np.array(voc_dictLabels[voc_cls])[voc_indexDtest].tolist())

            # shuffle the correponding relation between support set and query set
            random.shuffle(support_x)  # 5*1
            random.shuffle(query_x)    # 5*15

            self.support_x_batch.append(support_x)  # append set to current sets, 10000*5*1
            self.query_x_batch.append(query_x)  # append sets to current sets, 10000*5*15

            support_y = []
            query_y = []

            for id in support_x:
                support_y_init = []
                image_bboxs = voc_dictImages[id]
                for image_bbox in image_bboxs:
                    class_idx = image_bbox[-1]
                    if class_idx in voc_select_cls:
                        selected_img = voc_select_cls.tolist()
                        image_bbox[-1] = selected_img.index(class_idx)
                        support_y_init.append(image_bbox)
                support_y.append(np.array(support_y_init))
                # support_y_np = np.array(support_y)

            for id in query_x:
                query_y_init = []
                image_bboxs = voc_dictImages[id]
                for image_bbox in image_bboxs:
                    class_idx = image_bbox[-1]
                    if class_idx in voc_select_cls:
                        image_bbox[-1] = voc_select_cls.tolist().index(class_idx)
                        query_y_init.append(image_bbox)
                query_y.append(np.array(query_y_init))
                # query_y_np = np.array(query_y)

            self.support_y_batch.append(support_y)  # append set to current sets, 10000*5*1
            self.query_y_batch.append(query_y)  # append sets to current sets, 10000*5*15
            # self.support_y_batch.append(support_y_np)  # append set to current sets, 10000*5*1
            # self.query_y_batch.append(query_y_np)  # append sets to current sets, 10000*5*15

    def get_data(self, batch_x, batch_y, index, size):
        x = torch.FloatTensor(size, 3, self.resize, self.resize)
        y = []

        for i, img_id in enumerate(batch_x[index]):
            img = Image.open(os.path.join(self.root, 'JPEGImages', img_id)).convert('RGB')
            x[i] = self.transform(img)
            w, h = img.size
            # boxes augment
            boxes = batch_y[index][i]
            # scale_w, scale_h = self.resize / w, self.resize / h
            # nw, nh = int(w * scale_w), int(h * scale_h)
            # boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w  # adjust box to resized img
            # boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h
            # box_w = boxes[:, 2] - boxes[:, 0]
            # box_h = boxes[:, 3] - boxes[:, 1]
            # boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            # boxes_xy = (boxes[..., 0:2] + boxes[..., 2:4]) // 2
            # boxes_wh = boxes[..., 2:4] - boxes[..., 0:2]
            # boxes[..., 0:2] = boxes_xy / self.resize
            # boxes[..., 2:4] = boxes_wh / self.resize
            y.append(torch.from_numpy(boxes))

        return x, y

    def __getitem__(self, index):
        support_x, support_y = self.get_data(self.support_x_batch, self.support_y_batch, index, self.set_size)
        query_x, query_y = self.get_data(self.query_x_batch, self.query_y_batch, index, self.query_size)
        return support_x, support_y, query_x, query_y

    def collate_fn(self, batch):
        support_x, support_y, query_x, query_y = [], [], [], []
        for spt_x, spt_y, qry_x, qry_y in batch:
            support_x.append(spt_x)
            query_x.append(qry_x)
            support_y.append(spt_y)
            query_y.append(qry_y)
        return support_x, support_y, query_x, query_y

    def __len__(self):
        # as we have built up to batch_size of sets, you can sample some small batch size of sets.
        return self.batch_size


if __name__ == '__main__':
    anchor = [[[116, 90], [156, 198], [373, 326]],
                [[30, 61], [62, 45], [59, 119]],
                [[10, 13], [16, 30], [33, 23]]]
    anchor = np.array(anchor).reshape((-1, 2))
    voc_path = '../VOC2012/'
    start_time = time.time()
    voc = VOC(voc_path, mode='train', n_way=5, k_shot=1, k_query=1, batch_size=100, resize=224, anchor=anchor)
    end_time = time.time()
    print(end_time-start_time)
    dataloader = DataLoader(voc, 5, True, num_workers=4, collate_fn=voc.collate_fn)
    # print(dataloader.dataset[0])
    for i, (spt_x, spt_y, qry_x, qry_y) in enumerate(dataloader):
        print(i)


