import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .utils import get_file_names, read_video_frames, video2patches

class VideoDataset(Dataset):
    def __init__(self, videos, labels):
        self.videos = videos
        self.labels = labels

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        return self.videos[idx].float(), torch.tensor(self.labels[idx], dtype=torch.long)


def load_data(dataset, window_size, patch_size=(32, 32)):
    if dataset == "DF":
        real_dirs = ["data/FaceForensics++/original_sequences/actors/c23",
                     "data/FaceForensics++/original_sequences/youtube/c23"]
        fake_dirs = ["data/FaceForensics++/manipulated_sequences/Deepfakes/c23"]
    elif dataset == "F2F":
        real_dirs = ["data/FaceForensics++/original_sequences/actors/c23",
                     "data/FaceForensics++/original_sequences/youtube/c23"]
        fake_dirs = ["data/FaceForensics++/manipulated_sequences/Face2Face/c23"]
    elif dataset == "FS":
        real_dirs = ["data/FaceForensics++/original_sequences/actors/c23",
                     "data/FaceForensics++/original_sequences/youtube/c23"]
        fake_dirs = ["data/FaceForensics++/manipulated_sequences/FaceSwap/c23"]
    elif dataset == "NT":
        real_dirs = ["data/FaceForensics++/original_sequences/actors/c23",
                     "data/FaceForensics++/original_sequences/youtube/c23"]
        fake_dirs = ["data/FaceForensics++/manipulated_sequences/NeuralTextures/c23"]
    else:
        real_dirs, fake_dirs = [], []

    videos_pos, videos_neg, labels = [], [], []

    for real_dir in real_dirs:
        frame_root = os.path.join(real_dir, "frames")
        for video_name in get_file_names(frame_root):
            clips = read_video_frames(os.path.join(frame_root, video_name), window_size)
            if not clips:
                continue
            patches = [video2patches(clip, *patch_size) for clip in clips]
            videos_pos.append(patches)
            labels.append([1] * len(patches))

    for fake_dir in fake_dirs:
        frame_root = os.path.join(fake_dir, "frames")
        for video_name in get_file_names(frame_root):
            clips = read_video_frames(os.path.join(frame_root, video_name), window_size)
            if not clips:
                continue
            patches = [video2patches(clip, *patch_size) for clip in clips]
            videos_neg.append(patches)
            labels.append([0] * len(patches))

    return videos_pos, videos_neg, labels

class VideoDataset(Dataset):
    def __init__(self, videos, labels, transform=None):
        self.videos = videos
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        label = self.labels[idx]
        if self.transform:
            video = self.transform(video)
        return video, label


def load_dataloader(dataset_name, batch_size=8, num_workers=4,
                    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                    patch_size=(32, 32), window_size=8):
    
    print(f"{dataset_name} data preprocessing...")

    videos_pos, videos_neg, labels = load_data(dataset_name, window_size, patch_size)

    v_labels_pos = torch.ones(len(videos_pos), dtype=torch.int64)
    v_labels_neg = torch.zeros(len(videos_neg), dtype=torch.int64)
    v_labels = torch.cat((v_labels_pos, v_labels_neg), dim=0)
    videos = videos_pos + videos_neg

    indices = list(range(len(v_labels)))
    labels_list = v_labels.tolist()

    train_idx, temp_idx, _, temp_labels = train_test_split(
        indices, labels_list, test_size=(1 - train_ratio),
        stratify=labels_list, random_state=42
    )

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, temp_labels, test_size=(1 - val_ratio_adjusted),
        stratify=temp_labels, random_state=42
    )

    train_videos = [videos[i] for i in train_idx]
    val_videos = [videos[i] for i in val_idx]
    test_videos = [videos[i] for i in test_idx]

    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    test_labels = [labels[i] for i in test_idx]

    train_videos = [clip for video in train_videos for clip in video]
    val_videos = [clip for video in val_videos for clip in video]
    test_videos = [clip for video in test_videos for clip in video]

    train_labels = [label for video in train_labels for label in video]
    val_labels = [label for video in val_labels for label in video]
    test_labels = [label for video in test_labels for label in video]
    
    train_loader = DataLoader(VideoDataset(train_videos, train_labels), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(VideoDataset(val_videos, val_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(VideoDataset(test_videos, test_labels), batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_dataloader(
        "NT", batch_size=16, num_workers=4,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
        patch_size=(32, 32), window_size=8
    )

    for videos_batch, labels_batch in train_loader:
        print(videos_batch.shape, labels_batch.shape)
        break