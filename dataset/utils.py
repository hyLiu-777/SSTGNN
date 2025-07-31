import os
import cv2
import numpy as np
import torch


def frame2Dpatches(F,h,w):

    nH = (F.shape[0]-1)//h+1
    
    nW = (F.shape[1]-1)//w+1
    
    N = nH*nW
    
    patches = None
    for i in range(nH):
        for j in range(nW):
            patch = torch.tensor(F[i*h:min((i+1)*h,F.shape[0]), j*w:min((j+1)*w,F.shape[1]),:])
            if patches is None:
                patches = torch.unsqueeze(patch,dim=0)
            else:
                patches = torch.cat([patches,torch.unsqueeze(patch,dim=0)],dim=0)
    patches=patches.permute(0,3,1,2)     
    return patches


def video2patches(V,h,w):
    nH = (V.shape[1]-1)//h+1
    nW = (V.shape[2]-1)//w+1
    N = nH*nW
    T = V.shape[0]
    patches = []
    for t in range(T):
        patches.append(torch.unsqueeze(frame2Dpatches(V[t],h,w),dim=0))
    patches = torch.cat(patches,dim=0)
    # len(patches)
    return patches

def get_file_names(folder_path):
    file_names = []
    for filename in os.listdir(folder_path):
        file_names.append(filename)
    return file_names


def read_video_frames(folder_path, window_len):
    frame_names = sorted(get_file_names(folder_path))  # 按照文件名排序
    all_frames = []

    # 读取并resize所有有效的图像
    for name in frame_names:
        img_path = os.path.join(folder_path, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        resized = cv2.resize(img, (224, 224))
        # if np.all(resized == 0):
        #     print(f"跳过全 0 图像: {img_path}")
        #     continue
        all_frames.append(resized)

    # 不足8帧直接返回空列表
    if len(all_frames) < 8:
        return []

    # 滑动窗口提取所有连续8帧序列
    windows = []
    for i in range(0, len(all_frames) - 8 + 1, window_len):
        window = all_frames[i:i+8]
        windows.append(np.array(window, dtype=np.float32))
        if len(windows)>=5:
            return windows

    return windows



def load_mask(image_path, threshold=128):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image,(224,224))
    _, mask = cv2.threshold(image, threshold, 1, cv2.THRESH_BINARY)
    return mask

def read_mask(folder_path):
    mask_names = get_file_names(folder_path)
    masks = []
    
    for name in mask_names:
        mask = load_mask(folder_path+"/"+name)
        masks.append(mask)
        if len(masks)>=8:
            break
    while len(masks)<8:
        masks.append(masks[-1])
    
    masks = np.array(masks, dtype=np.float32)  
    masks = torch.from_numpy(masks)
    return masks
