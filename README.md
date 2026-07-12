# GraspNet Baseline
Baseline model for "GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping" (CVPR 2020).

[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fang_GraspNet-1Billion_A_Large-Scale_Benchmark_for_General_Object_Grasping_CVPR_2020_paper.pdf)]
[[dataset](https://graspnet.net/)]
[[API](https://github.com/graspnet/graspnetAPI)]
[[doc](https://graspnetapi.readthedocs.io/en/latest/index.html)]

<div align="center">    
    <img src="https://github.com/chenxi-wang/materials/blob/master/graspnet-baseline/doc/gifs/scene_0114.gif", width="240", alt="scene_0114" />
    <img src="https://github.com/chenxi-wang/materials/blob/master/graspnet-baseline/doc/gifs/scene_0116.gif", width="240", alt="scene_0116" />
    <img src="https://github.com/chenxi-wang/materials/blob/master/graspnet-baseline/doc/gifs/scene_0117.gif", width="240", alt="scene_0117" />
    <br> Top 50 grasps detected by our baseline model.
</div>

![teaser](doc/teaser.png)

## Requirements
- Python 3
- PyTorch 1.6
- Open3d >=0.8
- TensorBoard 2.3
- NumPy
- SciPy
- Pillow
- tqdm

## Installation
Get the code.
```bash
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

## Tolerance Label Generation
Tolerance labels are not included in the original dataset, and need additional generation. Make sure you have downloaded the orginal dataset from [GraspNet](https://graspnet.net/). The generation code is in [dataset/generate_tolerance_label.py](dataset/generate_tolerance_label.py). You can simply generate tolerance label by running the script: (`--dataset_root` and `--num_workers` should be specified according to your settings)
```bash
cd dataset
sh command_generate_tolerance_label.sh
```

Or you can download the tolerance labels from [Google Drive](https://drive.google.com/file/d/1DcjGGhZIJsxd61719N0iWA7L6vNEK0ci/view?usp=sharing)/[Baidu Pan](https://pan.baidu.com/s/1HN29P-csHavJF-R_wec6SQ) and run:
```bash
mv tolerance.tar dataset/
cd dataset
tar -xvf tolerance.tar
```

## Training and Testing
Training examples are shown in [command_train.sh](command_train.sh). `--dataset_root`, `--camera` and `--log_dir` should be specified according to your settings. You can use TensorBoard to visualize training process.

Testing examples are shown in [command_test.sh](command_test.sh), which contains inference and result evaluation. `--dataset_root`, `--camera`, `--checkpoint_path` and `--dump_dir` should be specified according to your settings. Set `--collision_thresh` to -1 for fast inference.

The pretrained weights can be downloaded from:

- `checkpoint-rs.tar`
[[Google Drive](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view?usp=sharing)]
[[Baidu Pan](https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A)]
- `checkpoint-kn.tar`
[[Google Drive](https://drive.google.com/file/d/1vK-d0yxwyJwXHYWOtH1bDMoe--uZ2oLX/view?usp=sharing)]
[[Baidu Pan](https://pan.baidu.com/s/1QpYzzyID-aG5CgHjPFNB9g)]

`checkpoint-rs.tar` and `checkpoint-kn.tar` are trained using RealSense data and Kinect data respectively.

## Demo
A demo program is provided for grasp detection and visualization using RGB-D images. You can refer to [command_demo.sh](command_demo.sh) to run the program. `--checkpoint_path` should be specified according to your settings (make sure you have downloaded the pretrained weights, we recommend the realsense model since it might transfer better). The output should be similar to the following example:

<div align="center">    
    <img src="doc/example_data/demo_result.png", width="480", alt="demo_result" />
</div>

__Try your own data__ by modifying `get_and_process_data()` in [demo.py](demo.py). Refer to [doc/example_data/](doc/example_data/) for data preparation. RGB-D images and camera intrinsics are required for inference. `factor_depth` stands for the scale for depth value to be transformed into meters. You can also add a workspace mask for denser output.

## Results
Results "In repo" report the model performance with single-view collision detection as post-processing. In evaluation we set `--collision_thresh` to 0.01.

Evaluation results on RealSense camera:
|          |        | Seen             |                  |        | Similar          |                  |        | Novel            |                  | 
|:--------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|
|          | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> |
| In paper | 27.56  | 33.43            | 16.95            | 26.11  | 34.18            | 14.23            | 10.55  | 11.25            | 3.98             |
| In repo  | 47.47  | 55.90            | 41.33            | 42.27  | 51.01            | 35.40            | 16.61  | 20.84            | 8.30             |

Evaluation results on Kinect camera:
|          |        | Seen             |                  |        | Similar          |                  |        | Novel            |                  | 
|:--------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|
|          | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> |
| In paper | 29.88  | 36.19            | 19.31            | 27.84  | 33.19            | 16.62            | 11.51  | 12.92            | 3.56             |
| In repo  | 42.02  | 49.91            | 35.34            | 37.35  | 44.82            | 30.40            | 12.17  | 15.17            | 5.51             |

## Citation
Please cite our paper in your publications if it helps your research:
```
@article{fang2023robust,
  title={Robust grasping across diverse sensor qualities: The GraspNet-1Billion dataset},
  author={Fang, Hao-Shu and Gou, Minghao and Wang, Chenxi and Lu, Cewu},
  journal={The International Journal of Robotics Research},
  year={2023},
  publisher={SAGE Publications Sage UK: London, England}
}

@inproceedings{fang2020graspnet,
  title={GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping},
  author={Fang, Hao-Shu and Wang, Chenxi and Gou, Minghao and Lu, Cewu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition(CVPR)},
  pages={11444--11453},
  year={2020}
}
```

# TODO

1. Statically validate the grasp position before the arm ever moves (blocker). We removed the empirical +0.09 Y offset with nothing replacing it, and the mirror correction moved from code into TF. Both need one ground-truth check: hold an object at a spot you can measure relative to the robot base (tape measure to panda_link0 origin), trigger inference, and compare ros2 topic echo /selected_grasp (after the orchestrator's transform — or add a Pose display in RViz with fixed frame panda_link0) against reality. If it's off by a consistent ~9 cm in one axis, that's the residual calibration error the old offset was masking — recalibrate before any motion. Five minutes of checking versus a 9 cm miss next to someone's hand.
---

# H2R Handover Pipeline — Usage

Full command reference for running the hand-to-robot handover pipeline
(RealSense D435 → EgoHOS segmentation → GraspNet → grasp selection →
MoveIt / Franka execution).

The `handover_orchestrator` is the single operator console — per handover
you press **Enter once to capture** a grasp and then **Enter to confirm each
planned motion**. It reads keyboard input, so it **must** run in its own
terminal — `ros2 launch` does not forward stdin. The two perception drivers
run inside their own venvs and also get their own terminals.

## One-time / after changing h2r_handovers

```bash
cd ~/projects/handovers_ws/handover_ws
colcon build --packages-select h2r_handovers
```

(The EgoHOS and GraspNet drivers run straight from source — no build needed,
just restart them.)

## Terminal 1 — camera, MoveIt + RViz, static TFs, grasp selection

```bash
source ~/projects/handovers_ws/handover_ws/install/setup.bash
ros2 launch h2r_handovers handover.launch.xml
```

## Terminal 2 — EgoHOS segmentation driver (egohos_venv)

```bash
source /opt/ros/humble/setup.bash
source ~/projects/handovers_ws/src/egohos_venv/bin/activate
python ~/projects/handovers_ws/src/EgoHOS/scripts/driver.py
```

## Terminal 3 — GraspNet driver (graspnet_venv)

```bash
source /opt/ros/humble/setup.bash
source ~/projects/handovers_ws/src/graspnet_venv/bin/activate
python ~/projects/handovers_ws/src/graspnet-baseline/scripts/driver.py
```

## Terminal 4 — handover orchestrator (operator console)

```bash
ros2 run h2r_handovers handover_orchestrator --ros-args \
  --params-file ~/projects/handovers_ws/handover_ws/src/h2r_handovers/config/handover_params.yaml
```

This terminal drives the whole handover:

- When the state is `IDLE`, it prompts `press Enter to capture a grasp` —
  hold the object steady in view and press **Enter**. This publishes the
  capture trigger that makes the GraspNet driver run inference.
- With `confirm_before_execute: true` (default), every arm motion is then
  planned first; inspect the trajectory in RViz (Planned Path display) and
  press **Enter** to execute it, or `n` + Enter to abort the handover.
- With `require_stable_hand: true` (and `hand_stabilization_node` running,
  currently commented out in the launch file), an armed capture waits for
  the `hand_stable` status before triggering.

## Typical test sequence

1. Start terminals 1–4; wait for "EgoHOS models loaded" and "GraspNet
   loaded" in terminals 2–3, and confirm the startup homing in terminal 4.
2. Hold the object in view of the camera, press Enter in terminal 4 at the
   `IDLE` prompt.
3. Watch `grasp_debug_image` (e.g. in rqt or RViz) — grasp candidates drawn
   over the object.
4. Still in terminal 4: pre-grasp is planned → check RViz → Enter → final
   approach is planned → check → Enter → gripper closes → release the object
   → confirm the remaining motions (retreat, dropoff, return home).
5. Back at `IDLE`, press Enter for the next handover.

## Useful debug topics

| Topic                    | Content                                        |
|--------------------------|------------------------------------------------|
| `system_state`           | Orchestrator state machine (latched), incl. the |
|                          | `WAITING_FOR_HAND` / `CAPTURING` phases        |
| `segmentation_overlay`   | EgoHOS hand/object mask over the RGB image     |
| `grasp_debug_image`      | GraspNet candidates drawn on the image         |
| `grasp_candidates`       | PoseArray of scored grasps (camera frame)      |
| `selected_grasp`         | PoseStamped chosen by the selection policy     |
| `capture_trigger`        | Empty msg, orchestrator → GraspNet driver      |
| `hand_stability_status`  | `no_hand` / `hand_unstable` / `hand_stable`    |

Saved debug frames (inputs + candidates) land in `/tmp/graspnet_inputs/`.
