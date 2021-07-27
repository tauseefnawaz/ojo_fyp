import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, directory, batch_size=1, shuffle=True, data_augmentation=True):
        self.batch_size = batch_size
        self.directory = directory
        self.shuffle = shuffle
        self.data_aug = data_augmentation
        self.X_path = self.search_data()
        self.print_stats()
        return None

    def search_data(self):
        X_path = []
        Y_dict = {}
        # list all kinds of sub-folders
        self.dirs = sorted(os.listdir(self.directory))
        for i, folder in enumerate(self.dirs):
            folder_path = os.path.join(self.directory, folder)
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                # append the each file path, and keep its label  
                X_path.append(file_path)
        return X_path

    def print_stats(self):
        # calculate basic information
        self.n_files = len(self.X_path)
        self.n_classes = len(self.dirs)
        self.indexes = np.arange(len(self.X_path))
        np.random.shuffle(self.indexes)
        # Output states
        print("Found {} files belonging to {} classes.".format(self.n_files, self.n_classes))
        return None

    def __len__(self):
        # calculate the iterations of each epoch
        steps_per_epoch = np.ceil(len(self.X_path) / float(self.batch_size))
        return int(steps_per_epoch)

    def __getitem__(self, index):
        # get the indexs of each batch
        batch_indexs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # using batch_indexs to get path of current batch
        batch_path = [self.X_path[k] for k in batch_indexs]
        # get batch data
        batch_x = self.data_generation(batch_path)
        return batch_x

    def on_epoch_end(self):
        # shuffle the data at each end of epoch
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_path):
        # load data into memory, you can change the np.load to any method you want
        batch_x = [self.load_data(x) for x in batch_path]
        # transfer the data format and take one-hot coding for labels
        batch_x = np.array(batch_x)
        return batch_x

    def normalize(self, data):
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / std

    def uniform_sampling(self, video, target_frames=64):
        # get total frames of input video and calculate sampling interval 
        len_frames = int(len(video))
        interval = int(np.ceil(len_frames / target_frames))
        # init empty list for sampled video and 
        sampled_video = []
        for i in range(0, len_frames, interval):
            sampled_video.append(video[i])
            # calculate numer of padded frames and fix it
        num_pad = target_frames - len(sampled_video)
        padding = []
        if num_pad > 0:
            for i in range(-num_pad, 0):
                try:
                    padding.append(video[i])
                except:
                    padding.append(video[0])
            sampled_video += padding
            # get sampled video
        return np.array(sampled_video, dtype=np.float32)

    def load_data(self, path):
        # load the processed .npy files which have 5 channels (1-3 for RGB, 4-5 for optical flows)
        data = np.load(path, mmap_mode='r')
        data = np.float32(data)
        # sampling 64 frames uniformly from the entire video
        data = self.uniform_sampling(video=data, target_frames=64)
        # normalize rgb images and optical flows, respectively
        data[..., :3] = self.normalize(data[..., :3])
        data[..., 3:] = self.normalize(data[..., 3:])

        return data


def getOpticalFlow(video):
    gray_video = []
    for i in range(len(video)):
        img = cv2.cvtColor(video[i], cv2.COLOR_RGB2GRAY)
        gray_video.append(np.reshape(img, (224, 224, 1)))

    flows = []
    for i in range(0, len(video) - 1):
        flow = cv2.calcOpticalFlowFarneback(gray_video[i], gray_video[i + 1], None, 0.5, 3, 15, 3, 5, 1.2,
                                            cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
        flow[..., 0] -= np.mean(flow[..., 0])
        flow[..., 1] -= np.mean(flow[..., 1])
        flow[..., 0] = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX)
        flow[..., 1] = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX)
        flows.append(flow)
    flows.append(np.zeros((224, 224, 2)))
    return np.array(flows, dtype=np.float32)


def Video2Npy(file_path, resize=(224, 224)):
    cap = cv2.VideoCapture(file_path)
    len_frames = int(cap.get(7))
    try:
        frames = []
        for i in range(len_frames - 1):
            _, frame = cap.read()
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.reshape(frame, (224, 224, 3))
            frames.append(frame)
    except:
        print("Error: ", file_path, len_frames, i)
    finally:
        frames = np.array(frames)
        cap.release()

    # Get the optical flow of video
    flows = getOpticalFlow(frames)

    result = np.zeros((len(flows), 224, 224, 5))
    result[..., :3] = frames
    result[..., 3:] = flows

    return result


def Save2Npy(file_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    videos = os.listdir(file_dir)
    for v in tqdm(videos):
        video_name = v.split('.')[0]
        video_path = os.path.join(file_dir, v)
        save_path = os.path.join(save_dir, video_name + '.npy')
        # Load and preprocess video
        data = Video2Npy(file_path=video_path, resize=(224, 224))
        data = np.uint8(data)
        np.save(save_path, data)




sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model = load_model('my_model.h5')

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def prediction(src_path, target_path, preprocessedDataPath):
    Save2Npy(src_path, target_path)
    test_generator = DataGenerator(directory=preprocessedDataPath,
                                   batch_size=1,
                                   data_augmentation=False)
    predict_prob = model.predict(test_generator[0])
    if predict_prob[0][0] > predict_prob[0][1]:
        return "True"
    return "False"

