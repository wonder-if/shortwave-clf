import librosa
from scipy.io import wavfile
import json
import os
import numpy as np


def read_mp3(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        return audio, sr
    except Exception as e:
        print(f"Error reading audio file {file_path}: {e}")
        return None, None


def read_wav(file_path, target_sr=16000):
    try:
        sr, wave = wavfile.read(file_path)
        return wave, sr
    except Exception as e:
        print(f"Error reading audio file {file_path}: {e}")


def normalize_signal(audio):
    """
    对音频信号进行归一化处理
    :param audio: 音频数据的 numpy 数组
    :return: 归一化后的音频数据
    """
    min_val = np.min(audio)
    max_val = np.max(audio)
    return (audio - min_val) / (max_val - min_val) * 2 - 1  # 归一化到 [-1, 1]


def src2proc(dataset_info_config_path="shortwave_clf/configs/dataset_info.json"):

    with open(dataset_info_config_path, "r") as f:
        dataset_info = json.load(f)
    src_data_dir = dataset_info["root"] + dataset_info["src_dir"]
    proc_data_dir = dataset_info["root"] + dataset_info["proc_dir"]
    print(f"Loading data from {src_data_dir}")

    class_names = os.listdir(src_data_dir)
    for class_name in class_names:
        class_dir = os.path.join(src_data_dir, class_name)
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            if file_name.endswith(".mp3"):
                audio, sr = read_mp3(file_path)
                print(f"Loaded {file_name} with sr={sr}, shape={audio.shape}")
                train_ratio = 0.6
                split_point = int(len(audio) * train_ratio)
                train_audio = audio[:split_point]
                test_audio = audio[split_point:]

                # 创建保存目录
                train_dir = os.path.join(proc_data_dir, "train")
                test_dir = os.path.join(proc_data_dir, "test")
                os.makedirs(train_dir, exist_ok=True)
                os.makedirs(test_dir, exist_ok=True)

                # 保存训练集和测试集
                train_file = f"{file_name.replace('.mp3', '_train.npy')}"
                test_file = f"{file_name.replace('.mp3', '_test.npy')}"

                if not os.path.exists(os.path.join(train_dir, class_name)):
                    os.makedirs(os.path.join(train_dir, class_name))
                if not os.path.exists(os.path.join(test_dir, class_name)):
                    os.makedirs(os.path.join(test_dir, class_name))

                np.save(os.path.join(train_dir, class_name, train_file), train_audio)
                np.save(os.path.join(test_dir, class_name, test_file), test_audio)

                print(f"Saved train set: {train_file}, shape: {train_audio.shape}")
                print(f"Saved test set: {test_file}, shape: {test_audio.shape}")


def extract_clips(audio, clip_length=1024, num_clips=10000):
    clips = []
    for i in range(num_clips):
        start = np.random.randint(0, len(audio) - clip_length)
        clip = audio[start : start + clip_length]
        clips.append(clip)
    return clips


def prec2raw(
    dataset_info_config_path="shortwave_clf/configs/dataset_info.json",
    clip_length=4096,
    num_clips=200000,
):
    with open(dataset_info_config_path, "r") as f:
        dataset_info = json.load(f)
    for tag in ["train", "test"]:
        proc_dir = dataset_info["root"] + dataset_info["proc_dir"] + f"{tag}/"
        raw_dir = dataset_info["root"] + dataset_info["dir"] + f"{tag}/"
        print(f"Loading processed data from {proc_dir}")

        if not os.path.exists(raw_dir):
            os.makedirs(raw_dir)

        class_names = os.listdir(proc_dir)
        dataset_clips = []
        dataset_labels = []
        for i in range(len(class_names)):
            class_dir = os.path.join(proc_dir, class_names[i])
            num_file = len(os.listdir(class_dir))
            num_clips_per_file = num_clips // num_file
            cls_clips = []
            cls_labels = [i] * (num_clips_per_file * num_file)
            for file_name in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file_name)

                audio = np.load(file_path)

                # normlization
                audio = normalize_signal(audio)

                # extract signals clips from audio
                print(f"Loaded {file_name} with shape={audio.shape}")
                clips = extract_clips(audio, clip_length, num_clips_per_file)
                cls_clips.extend(clips)

            dataset_clips.extend(cls_clips)
            dataset_labels.extend(cls_labels)
        dataset_clips = np.array(dataset_clips)
        dataset_labels = np.array(dataset_labels)
        print(f"Dataset shape: {dataset_clips.shape}")
        print(f"Dataset labels shape: {dataset_labels.shape}")
        np.save(os.path.join(raw_dir, "signals.npy"), dataset_clips)
        np.save(os.path.join(raw_dir, "labels.npy"), dataset_labels)


if __name__ == "__main__":
    # src2proc()
    prec2raw()
    pass
