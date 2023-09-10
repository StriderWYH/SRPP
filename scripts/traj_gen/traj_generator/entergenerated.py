import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_bezier_curve(start, control, end, num_points=160):
    curve = []
    for t in np.linspace(0, 1, num_points):
        point = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end
        curve.append(point)
    return np.array(curve)

def check_distance(points, threshold=800):
    for point in points:
        if np.linalg.norm(point) > threshold:
            return False
    return True

def save_to_txt(curves, filename='bezier_curves.txt'):
    with open(filename, 'w') as f:
        for curve in curves:
            np.savetxt(f, curve)
            f.write("---\n")

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

curves = load_from_txt()
num_curves = 30
num_curves_done = len(curves)

while num_curves_done < num_curves:
    terminate = input("你想终止程序吗？（yes/no）: ")
    if terminate.lower() == 'yes':
        save_to_txt(curves)
        print(f"{len(curves)}条曲线已保存到 'bezier_curves.txt'")
        break
    
    try:
        start = np.array(list(map(float, input(f"请输入第 {num_curves_done+1} 条曲线的起点（格式：x y z）: ").split())))
        control = np.array(list(map(float, input(f"请输入第 {num_curves_done+1} 条曲线的控制点（格式：x y z）: ").split())))
        end = np.array(list(map(float, input(f"请输入第 {num_curves_done+1} 条曲线的终点（格式：x y z）: ").split())))

        if start.shape[0] != 3 or control.shape[0] != 3 or end.shape[0] != 3:
            print("输入的点应该是三维的。")
            continue

        curve = generate_bezier_curve(start, control, end)

        if not check_distance(curve):
            print("有点到原点的距离超过800，重新输入三个点。")
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(curve[:, 0], curve[:, 1], curve[:, 2], marker='o')
        plt.show()

        save_or_not = input("保存这条曲线吗？ (yes/no): ")

        if save_or_not.lower() == 'yes':
            curves.append(curve)
            num_curves_done += 1

        plt.close()

    except Exception as e:
        print(f"出现错误：{e}")
        print("请确保输入符合规范，重新输入。")

    if num_curves_done >= num_curves:
        save_to_txt(curves)
        print(f"{num_curves}条曲线已保存到 'bezier_curves.txt'")
        break
