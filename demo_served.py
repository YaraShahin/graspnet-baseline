import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import time
import copy
from skimage.io import imread
import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

CHECKPOINT_PATH = r"/home/yara/camera_ws/src/graspnet-baseline/checkpoint-rs.tar"
num_points = 20000
num_view = 300

COLOR_PATH = r"/home/yara/camera_ws/src/graspnet-baseline/doc/example_data/color.png"
DEPTH_PATH = r"/home/yara/camera_ws/src/graspnet-baseline/doc/example_data/depth.png"
WORKSPACE_PATH = r"/home/yara/camera_ws/src/graspnet-baseline/doc/example_data/workspace_mask.png"
HAND_SEG_PATH = r"/home/yara/camera_ws/src/EgoHOS/testimages/pred_twohands/color.png"

OUTPUT_PATH = r"/home/yara/camera_ws/src/graspnet-baseline/doc/example_data/transformation_matrix.npy"
HAND_OUTPUT_PATH = r"/home/yara/camera_ws/src/graspnet-baseline/doc/example_data/hand_transformation_matrix.npy"
WIDTH_PATH = r"/home/yara/camera_ws/src/graspnet-baseline/doc/example_data/width.txt"

print("[GRASPNET] Initializing Model")
curr_time = time.time()
net = GraspNet(input_feature_dim=0, num_view=num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0")

# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH)

print("[GRASPNET] Done; Time:", time.time() -  curr_time)

def get_point_translation_matrix(centroid_x, centroid_y):
    fx, fy, cx, cy = 911.95, 912.27, 651.08, 347.59
    depth_img = np.array(Image.open(DEPTH_PATH))
    Z = depth_img[centroid_y, centroid_x]
    X = (centroid_x - cx) * Z / fx
    Y = (centroid_y - cy) * Z / fy
    return (X, Y, Z)

def get_centroid(img_path):
    image = imread(img_path)
    #print(image)
    # Convert the image to a NumPy array
    image_array = np.array(image)
    #print(image_array)
    
    # Find the coordinates of all white pixels (value 2)
    white_pixels = np.argwhere(image_array == 2)
    #print(white_pixels)
    
    # Calculate the mean of the x and y coordinates
    try:
        centroid_y, centroid_x = np.mean(white_pixels, axis=0)
    except:
        centroid_y, centroid_x = 0, 0
        print("[GRASPNET] ERROR: HAND Centroid FAILED!")
    
    return int(centroid_x), int(centroid_y)
    
def get_net():
    net_instance = copy.deepcopy(net)
    net_instance.to(device)
    net_instance.load_state_dict(checkpoint['model_state_dict'])
    net_instance.eval()
    return net_instance

def get_and_process_data():
    # load data
    color = np.array(Image.open(COLOR_PATH), dtype=np.float32) / 255.0
    depth = np.array(Image.open(DEPTH_PATH))
    workspace_mask = np.array(Image.open(WORKSPACE_PATH))
    
    fx, fy, cx, cy = 911.95, 912.27, 651.08, 347.59
    factor_depth = 1000

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, fx, fy, cx, cy, factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= num_points:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), num_points-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked
    color_sampled = color_masked

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    gg.nms()
    gg.sort_by_score()
    the_grasp = None
    current_depth = 70
    the_i = 0

    for i in range(0, 20):
        """if (gg[i].translation[2] < 0.5):
            the_grasp = gg[i]
            break"""

        if (gg[i].translation[2] < current_depth):
            the_grasp = gg[i]
            current_depth = gg[i].translation[2]
            the_i = i

    gg[the_i].score = 100000
    gg.sort_by_score()
    if ((not the_grasp) or (the_grasp.translation[2] > 0.5)):
        return 0, 0    
    
    return gg, the_grasp

def combine_translation_rotation(translation, rotation = np.zeros((3,3))):
    # Ensure translation is a column vector
    translation = np.array(translation).reshape(3, 1)
    
    # Create a 4x4 identity matrix
    transformation_matrix = np.eye(4)
    
    # Assign rotation matrix to the upper left 3x3 block
    transformation_matrix[:3, :3] = rotation
    
    # Assign translation to the last column
    transformation_matrix[:3, 3] = translation.flatten()
    
    return transformation_matrix

def vis_grasps(gg, cloud):
    gg = gg[:2]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo():
    net_instance = get_net()
    end_points, cloud = get_and_process_data()
    gg, the_grasp = get_grasps(net_instance, end_points)

    del net_instance
    torch.cuda.empty_cache()

    if (the_grasp == 0):
        return 0

    print("============= POSE GENERATION RESULTS ==============")
    print("Translation:", the_grasp.translation)
    print("Rotation:\n", the_grasp.rotation_matrix)
    rot_mat = combine_translation_rotation(the_grasp.translation, the_grasp.rotation_matrix)
    print("4x4 rotation matrix:\n", rot_mat)
    print("width:", the_grasp.width, "\n")

    print("============= HAND POSE RESULTS ==============")
    hand_centroid_x, hand_centroid_y = 0,0
    #hand_centroid_x, hand_centroid_y = get_centroid(HAND_SEG_PATH) 
    print("hii")
    hand_translation = get_point_translation_matrix(hand_centroid_x, hand_centroid_y)
    print("hii2")
    hand_rot_mat = combine_translation_rotation(hand_translation)
    print("4x4 rotation matrix:\n", hand_rot_mat)

    np.save(OUTPUT_PATH, rot_mat)
    np.save(HAND_OUTPUT_PATH, hand_rot_mat)

    with open(WIDTH_PATH, 'w') as file:
        file.write(str(the_grasp.width))
    
    vis_grasps(gg, cloud)
    return net

if __name__=='__main__':
    try:
        while True:
            if (os.path.exists(COLOR_PATH) and os.path.exists(DEPTH_PATH) and os.path.exists(WORKSPACE_PATH) and os.path.exists(HAND_SEG_PATH)):
                curr_time = time.time()
                print("[GRASPNET] Generating Grasp Pose...")
                demo()
                print("[GRASPNET] Generation done; Cleaning ws...")
                os.remove(COLOR_PATH)
                os.remove(DEPTH_PATH)
                os.remove(WORKSPACE_PATH)
                os.remove(HAND_SEG_PATH)
                print("[GRASPNET] Finally Done; Time:", time.time() -  curr_time)
            else:
                time.sleep(2)
                ##print(os.path.exists(COLOR_PATH) and os.path.exists(DEPTH_PATH) and os.path.exists(WORKSPACE_PATH))
                #print(os.path.exists(HAND_SEG_PATH))
    except Exception as e:
        print("error in graspnet", e)

