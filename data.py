import os.path
import numpy as np
import pandas as pd
import torch.utils.data


def load_training_data(yellowcloth_file_path, plastic_file_path, paper_file_path):
    # YELLOW CLOTHES
    cloth_data = pd.read_csv(yellowcloth_file_path, header=None)
    cloth_data = np.array(cloth_data)
    cloth_data = np.transpose(cloth_data)
    cloth_data = np.reshape(cloth_data, (16000, 1))
    # cloth_data = cloth_data.ravel()
    print(len(cloth_data))

    # PLASTIC
    plastic_data = pd.read_csv(plastic_file_path, header=None)
    plastic_data = np.array(plastic_data)
    plastic_data = np.transpose(plastic_data)
    plastic_data = np.reshape(plastic_data, (16000, 1))
    # print(plastic_data)

    # PAPER
    paper_data = pd.read_csv(paper_file_path, header=None)
    paper_data = np.array(paper_data)
    paper_data = np.transpose(paper_data)
    paper_data = np.reshape(paper_data, (16000, 1))
    # print(paper_data)

    return cloth_data, plastic_data, paper_data


# def load_training_data(num, sole_file_path, IMU_L_ank_file_path, IMU_R_ank_file_path, IMU_waist_file_path):
    # dir = os.getcwd()
    # sole_data = pd.read_table(sole_file_path % (dir, num), delimiter='\t', header=None, skiprows=12, usecols=[3, 5, 7, 9, 12, 14, 16, 18])
    # sole_data = np.array(sole_data)
    # IMU_L_ank = np.loadtxt(IMU_L_ank_file_path % (dir, num), delimiter=',', usecols=[2, 3, 4, 5, 6, 7])
    # IMU_R_ank = np.loadtxt(IMU_R_ank_file_path % (dir, num), delimiter=',', usecols=[2, 3, 4, 5, 6, 7])
    # IMU_waist = np.loadtxt(IMU_waist_file_path % (dir, num), delimiter=',', usecols=[2, 3, 4, 5, 6, 7])

    # sole sync data
    # sole_sync = np.argmax(sole_data[200:600, 7])
    # sole_sync_pozi = sole_sync + 100
    # sole = sole_data[sole_sync_pozi:, :]
    # sole_L = sole[:, :3]
    # sole_R = sole[:, 4:7]
    # sole = np.hstack([sole_L, sole_R])

    # IMU sync data
    # IMU_sinc = np.argmin(IMU_R_ank[200:600, 0])
    # IMU_sync_pozi = IMU_sinc + 100
    # IMU_L_ank = IMU_L_ank[IMU_sync_pozi:, :]
    # IMU_R_ank = IMU_R_ank[IMU_sync_pozi:, :]
    # IMU_waist = IMU_waist[IMU_sync_pozi:, :]

    # if len(IMU_L_ank) > len(IMU_R_ank):
    #     IMU_L_ank = IMU_L_ank[:len(IMU_R_ank), :]
    # else:
    #     IMU_R_ank = IMU_R_ank[:len(IMU_L_ank), :]
    # IMU_data = np.hstack([IMU_L_ank, IMU_R_ank])

    # if len(IMU_data) > len(IMU_waist):
    #     IMU_data = IMU_data[:len(IMU_waist), :]
    # else:
    #     IMU_waist = IMU_waist[:len(IMU_data), :]
    # IMU_data = np.hstack([IMU_data, IMU_waist])

    # # conbine
    # if len(sole) > len(IMU_data):
    #     sole = sole[:len(IMU_data), :]
    # else:
    #     IMU_data = IMU_data[:len(sole), :]

    # data = np.hstack([sole, IMU_data])
    # data_1 = data[250:750, :]
    # data_2 = data[1250:1750, :]
    # data_3 = data[2250:2750, :]

    # return data_1, data_2, data_3


class ActionDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.YELLOWCLOTH_PATH = './data/FVR/yellowCloth_train.csv'
        self.PLASTIC_PATH = './data/FVR/plastic_train.csv'
        self.PAPER_PATH = './data/FVR/paper_train.csv'
        # self.SOLE_PATH = '%s/data/sole/test%d.mva'
        # self.L_ANK_PATH = '%s/data/IMU/L_leg_%d.csv'
        # self.R_ANK_PATH = '%s/data/IMU/R_leg_%d.csv'
        # self.WAIST_PATH = '%s/data/IMU/Waist_%d.csv'

        self.sequences = []
        self.labels = []
        self.tags = []
        yellowCloth, plastic, paper = load_training_data(self.YELLOWCLOTH_PATH, self.PLASTIC_PATH, self.PAPER_PATH)
        # sit, stand, walk = load_training_data(data_num + 1, self.SOLE_PATH, self.L_ANK_PATH, self.R_ANK_PATH, self.WAIST_PATH)
        self.sequences.append(yellowCloth.astype(np.float32))
        self.labels.append(np.array([1, 0, 0]).astype(np.float32))
        self.tags.append(0)
        self.sequences.append(plastic.astype(np.float32))
        self.labels.append(np.array([0, 1, 0]).astype(np.float32))
        self.tags.append(1)
        self.sequences.append(paper.astype(np.float32))
        self.labels.append(np.array([0, 0, 1]).astype(np.float32))
        self.tags.append(2)

    def __len__(self):
        return self.labels.__len__()

    def __getitem__(self, index):
        return {
            'values': self.sequences[index],
            'label': self.labels[index],
            'raw_label': self.tags[index]
        }
