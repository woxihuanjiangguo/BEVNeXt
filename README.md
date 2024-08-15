# BEVNeXt
This is the official repository of BEVNeXt: Reviving Dense BEV Frameworks for 3D Object Detection.
[Paper link](https://arxiv.org/pdf/2312.01696).

# Installation & Dataset Preparation
Our repository is based on [BEVDet](https://github.com/HuangJunJie2017/BEVDet).
+ Please follow [docs/getting_started.md](docs/en/getting_started.md) to prepare the Anaconda environment.
+ Prepare the nuScenes dataset based on instructions in [docs/data_preparation.md](docs/en/data_preparation.md).
+ Run the script to generate pkl:
    ```
    python tools/create_data_bevdet.py
    ```

# Model Zoo
|          Backbone           |  Pretrain  |     Method     | NDS  |  mAP |                           Config                            | Download  |
|:---------------------------:|:----------:|:--------------:|:----:|-----:|:-----------------------------------------------------------:|:---------:|
|             R50             |  ImageNet  | BEVNeXt-Stage1 |  -   |    - |         [config](configs/bevnext/bevnext-stage1.py)         | [model](https://huggingface.co/Zzxxxxxxxx/bevnext/resolve/main/bevnext_stage1.pth?download=true) |
|             R50             |     -      | BEVNeXt-Stage2 | 54.8 | 43.7 |         [config](configs/bevnext/bevnext-stage2.py)         | [model](https://huggingface.co/Zzxxxxxxxx/bevnext/resolve/main/bevnext_stage2.pth?download=true) |
|             R50             | [Fcos3d](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth) | BEVNeXt-Stage1 |  -   |    - | [config](configs/bevnext/bevnext-pers-pretrained-stage1.py) | [model](https://huggingface.co/Zzxxxxxxxx/bevnext/resolve/main/bevnext_perspective_stage1.pth?download=true) |
|             R50             |     -      | BEVNeXt-Stage2 | 56.0 | 45.6 | [config](configs/bevnext/bevnext-pers-pretrained-stage2.py) | [model](https://huggingface.co/Zzxxxxxxxx/bevnext/resolve/main/bevnext_perspective_stage2.pth?download=true) |

# Training & Inference

+ Training-Stage1: This stage uses no temporal information to warm the model up, as is done in [SOLOFusion](https://github.com/Divadi/SOLOFusion).
```
# if R50 with perspective pretraining is to be used
# remember to download the Fcos3d checkpoint and fill in the path in bevnext-pers-pretrained-stage1.py
cfg="configs/bevnext/bevnext-stage1.py"
work_dir="work_dirs/bevnext-stage1"
bash tools/dist_train.sh $cfg 8 --work-dir $work_dir --seed 0
```
+ Training-Stage2 (Single Node): This stage loads the weights from the previous stage and uses long-term temporal information for training. The BEV Encoder and Detection Heads from the previous stage are discarded.
```
# remember to fill in the checkpoint path from the previous stage in bevnext-stage2.py
cfg="configs/bevnext/bevnext-stage2.py"
work_dir="work_dirs/bevnext-stage2"
bash tools/dist_train.sh $cfg 8 --work-dir $work_dir --seed 0
```
+ Training-Stage2 (Multi-Node): Obtaining historical features in a sliding window manner is generally slow. Using 16 gpus is recommended.
```
cfg="configs/bevnext/bevnext-stage2.py"
work_dir="work_dirs/bevnext-stage2"
NNODES=2 NODE_RANK=your_node_rank MASTER_ADDR=your_master_node_ip \
    bash tools/dist_train.sh $cfg 8 --work-dir $work_dir --seed 0
```
+ Inference
```
epoch_cnt=12
dir=your/path/to/ckpts
bash tools/dist_test.sh $dir/*.py $dir/epoch_${epoch_cnt}_ema.pth 8 --eval mAP --no-aavt
```
# Acknowledgements
This codebase is largely based on the [BEVDet Series](https://github.com/HuangJunJie2017/BEVDet). 
We also would like to thank the following repositories: 
+ [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
+ [SOLOFusion](https://github.com/Divadi/SOLOFusion)
+ [FB-BEV](https://github.com/NVlabs/FB-BEV)
+ [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
# Citation
```
@inproceedings{li2024bevnext,
  title={BEVNeXt: Reviving Dense BEV Frameworks for 3D Object Detection},
  author={Li, Zhenxin and Lan, Shiyi and Alvarez, Jose M and Wu, Zuxuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20113--20123},
  year={2024}
}
```