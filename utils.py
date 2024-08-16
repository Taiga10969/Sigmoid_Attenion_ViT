import numpy as np
import matplotlib.pyplot as plt


def Attention_Rollout(vision_attns):
    mean_head = np.mean(vision_attns, axis=1)
    mean_head = mean_head + np.eye(mean_head.shape[1])
    mean_head = mean_head / mean_head.sum(axis=(1,2))[:, np.newaxis, np.newaxis]

    v = mean_head[-1]
    for n in range(1,len(mean_head)):
        v = np.matmul(v, mean_head[-1-n])
    
    return v

def heatmap_to_rgb(heatmap, cmap='jet'):
    # 0から1の範囲に正規化
    normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
    # カラーマップに変換
    colormap = plt.get_cmap(cmap)
    colored_heatmap = (colormap(normalized_heatmap) * 255).astype(np.uint8)
    # RGBに変換
    rgb_image = colored_heatmap[:, :, :3]
    return rgb_image