import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

def load_from_txt(filename='bezier_curves.txt'):
    curves = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            curve = []
            for line in f:
                if "---" in line:
                    curves.append(np.array(curve))
                    curve = []
                else:
                    curve.append(list(map(float, line.strip().split())))
    return curves

def inverse_kinetics(list):
    return []

def cal_velocity(positions, dt):
    velocity = np.diff(positions, axis=0) / dt
    return np.vstack((velocity, velocity[-1]))

def cal_accleration(velocity, dt):
    acceleration = np.diff(velocity, axis=0) / dt
    return np.vstack((acceleration, acceleration[-1]))

# 计算曲率的函数
def cal_curvature(velocity, acceleration):
    numerator = np.linalg.norm(np.cross(velocity, acceleration), axis=1)
    denominator = np.linalg.norm(velocity, axis=1) ** 3 + 1e-10   # 防止除0
    curvature = numerator / denominator
    return curvature

def cal_jointNorm(joints):
    lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upper_limits = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    # 计算每个点与上下限的距离
    if joints == []:
        return 0
    lower_distances = joints - lower_limits
    upper_distances = upper_limits - joints

    # 求两者中的最小值，即每个点距离上下限的最小距离
    min_distances = np.minimum(lower_distances, upper_distances)

    # 求这些最小距离的范数
    return np.linalg.norm(min_distances, axis=1)




def segment_trajectory(X, window_size=20, step_size=10, n_clusters=3, agg_func=np.mean):
    """
    对轨迹进行分段。

    参数:
    X -- N x 1 向量，其中 N 是点的数量，每个点有一个综合得分
    window_size -- 滑动窗口的大小
    step_size -- 滑动窗口的步长
    n_clusters -- 聚类数量
    agg_func -- 用于窗口内特性聚合的函数（默认为 np.mean）

    返回:
    segments -- 分段的列表，每个分段是一个(start, end, label)的元组
    """
  
    # 滑动窗口
    n_windows = (len(X) - window_size) // step_size + 1
    windows = [X[i:i+window_size] for i in range(0, len(X) - window_size + 1, step_size)]

    # 窗口内特性聚合
    window_features = np.array([agg_func(window) for window in windows]).reshape(-1, 1)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters).fit(window_features)

    # 标记与分段
    labels = kmeans.labels_
    segments = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((start_idx, i-1, current_label))
            start_idx = i
            current_label = labels[i]
            
    segments.append((start_idx, len(labels)-1, current_label))
    
    return segments


def main(time_delta = 0.08, weight_velocity = 0, weight_curvature = 1, weight_accleration = 0, weight_jointLimit = 0, window_size=10, step_size=5, n_clusters=3):
    curves = load_from_txt()
    for curve in curves:
        # 计算关节角度
        joints = inverse_kinetics(curve) # 返回 N*7
        # 计算速度
        velocities = cal_velocity(curve, time_delta) # 返回 N*3
        # 计算加速度
        accelerations = cal_accleration(velocities, time_delta)  # 返回 N*3
        # 计算曲率
        curvature = cal_curvature(velocities, accelerations) # 返回 N*1
        
        velocitiesNorm = np.linalg.norm(velocities, axis=1) # 返回 N*1
        accelerationsNorm = np.linalg.norm(accelerations, axis=1) # 返回 N*1
        
        velocity_normalized = (velocitiesNorm - np.min(velocitiesNorm)) / (np.max(velocitiesNorm) - np.min(velocitiesNorm))
        acceleration_normalized = (accelerationsNorm - np.min(accelerationsNorm)) / (np.max(accelerationsNorm) - np.min(accelerationsNorm))
        curvature_normalized = (curvature - np.min(curvature)) / (np.max(curvature) - np.min(curvature))
        joint_angle_normalized = cal_jointNorm(joints)
        
        overall_score = velocity_normalized * weight_velocity + acceleration_normalized * weight_accleration + curvature_normalized * weight_curvature + joint_angle_normalized * weight_jointLimit

        # 进行分段
        segments = segment_trajectory(overall_score, window_size, step_size, n_clusters)

        # 创建 3D 图
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 可视化分段
        colors = ['r', 'g', 'b']  # 可以添加更多颜色，以匹配 n_clusters 的数量

        for start, end, label in segments:
            start_idx = start * step_size  # 调整这个以匹配你的实际步长
            end_idx = (end + 1) * step_size  # 调整这个以匹配你的实际窗口大小和步长
            segment_points = curve[start_idx:end_idx]

            ax.plot(segment_points[:, 0], segment_points[:, 1], segment_points[:, 2], c=colors[label % len(colors)])

        x_max, y_max, z_max = np.max(np.abs(curve), axis=0)

        # 设置轴的范围，使其以（0,0,0）为中心
        ax.set_xlim([-x_max, x_max])
        ax.set_ylim([-y_max, y_max])
        ax.set_zlim([-z_max, z_max])

        # 设置轴比例
        ax.set_aspect('equal', 'box')
        
        # 添加轴标签
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.title('Segmented 3D Trajectory')
        plt.show()


    
    
if __name__ == '__main__':
    main()