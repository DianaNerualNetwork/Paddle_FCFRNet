# 【PPDepth】 FCFRNet（AAAI2021）

A paddle implementation of the paper FCFRNet: Fusion based Coarse-to-Fine Residual Learning for Monocular Depth Completion [\[AAAI2021\]](https://arxiv.org/pdf/2012.08270v1.pdf)

# Abstract
Depth completion aims to recover a dense depth map from a sparse depth map with the corresponding color image as input. Recent approaches mainly formulate the depth completion as a one-stage end-to-end learning task, which outputs dense depth maps directly. However, the feature extraction and supervision in one-stage frameworks are insufficient,limiting the performance of these approaches. To address this problem, we propose a novel end-to-end residual learning framework, which formulates the depth completion as a twostage learning task, i.e., a sparse-to-coarse stage and a coarseto-fine stage. First, a coarse dense depth map is obtained by a simple CNN framework. Then, a refined depth map is further obtained using a residual learning strategy in the coarse-tofine stage with coarse depth map and color image as input. Specially, in the coarse-to-fine stage, a channel shuffle extraction operation is utilized to extract more representative features from color image and coarse depth map, and an energy
based fusion operation is exploited to effectively fuse these features obtained by channel shuffle operation, thus leading to more accurate and refined depth maps. We achieve SoTA performance in RMSE on KITTI benchmark. Extensive experiments on other datasets future demonstrate the superiority of our approach over current state-of-the-art depth completion approaches.

# Data prepare
This project use the preprocessed KITTI dataset.
KITTI DC dataset is available at the [KITTI DC Website](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion).

For color images, KITTI Raw dataset is also needed, which is available at the [KITTI Raw Website](http://www.cvlibs.net/datasets/kitti/raw_data.php).

Please follow the official instructions (cf., devkit/readme.txt in each dataset) for preparation.


In this project,we provide a second stage framework to do depth completion. Thus,you need do sparse to coarse  in a common network,and use it in this project. You can download  in [this]()


```
.
├── depth_selection
│    ├── test_depth_completion_anonymous
│    │    ├── image
│    │    ├── intrinsics
│    │    └── velodyne_raw
│    ├── test_depth_prediction_anonymous
│    │    ├── image
│    │    └── intrinsics
│    └── val_selection_cropped
│        ├── groundtruth_depth
|        |── data_depth_dense_new  #### Need to attention!
│        ├── image
│        ├── intrinsics
│        └── velodyne_raw
├── train
│    ├── 2011_09_26_drive_0001_sync
│    │    ├── image_02
│    │    │     └── data
│    │    ├── image_03
│    │    │     └── data
│    │    ├── oxts
│    │    │     └── data
│    │    └── proj_depth
│    │        ├── groundtruth
│    │        └── velodyne_raw
│    └── ...
└── val
    ├── 2011_09_26_drive_0002_sync
    └── ...
```

before u use this code, you need to change this path on your computer. 
## FCFRNet/dataloader/kitti_loader.py

```bash
def get_paths_and_transform(split, args):
    assert (args.use_d or args.use_rgb
            or args.use_g), 'no proper input selected'

    if split == "train":
        transform = train_transform
        glob_d = os.path.join(
            args.data_folder,
         # 'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')  ###########Note: This path is frist stage(sparse to coarse)result depth (dense depth result)
           'kitti_dc_4k/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')
        #'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        glob_gt = os.path.join(
            args.data_folder,
            'kitti_dc_4k/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'  ###########Note: This path is groundturth depth
        )

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([args.data_folder] + ['kitti_dc_4k'] + ps[-6:-4] +
                            ps[-2:-1] + ['data'] + ps[-1:])
            return pnew
    elif split == "val":
        if args.val == "full":
            transform = val_transform
            glob_d = os.path.join(
                args.data_folder,
              #'data_depth_velodyne/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
                  'kitti_dc_4k/val/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
            #)
             
            glob_gt = os.path.join(
                args.data_folder,
                'kitti_dc_4k/val/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
            )
            def get_rgb_paths(p):
                ps = p.split('/')
                pnew = '/'.join(ps[:-7] +  
                    ['kitti_dc_4k']+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
                return pnew
        elif args.val == "select":
        #    transform = train_transform
        #    glob_d = os.path.join(
        #        args.data_folder,
        # # 'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png')
       #        'data_depth_dense_new/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png')
       # #'data_depth_velodyne/train/*_sync/proj_depth/velodyne_raw/image_0[2,3]/*.png'
        #    glob_gt = os.path.join(
         #       args.data_folder,
         #       'data_depth_annotated/train/*_sync/proj_depth/groundtruth/image_0[2,3]/*.png'
          #  )

           # def get_rgb_paths(p):
           #     ps = p.split('/')
           #     pnew = '/'.join([args.data_folder] + ['data_rgb'] + ps[-6:-4] +
           #                 ps[-2:-1] + ['data'] + ps[-1:])
           #     return pnew
            transform = no_transform
            glob_d = os.path.join(
                args.data_folder,
                "kitti_dc_4k/depth_selection/val_selection_cropped/velodyne_raw/*.png")
                #"depth_selection/val_selection_cropped/velodyne_raw/*.png")
            glob_gt = os.path.join(
                args.data_folder,
                "kitti_dc_4k/depth_selection/val_selection_cropped/groundtruth_depth/*.png"
            )
            def get_rgb_paths(p):
                return p.replace("groundtruth_depth","image")
    elif split == "test_completion":
        transform = no_transform
        glob_d = os.path.join(
            args.data_folder,
            "kitti_dc_4k/depth_selection/test_depth_completion_anonymous/velodyne_raw/*.png"
        )
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "kitti_dc_4k/depth_selection/test_depth_completion_anonymous/image/*.png")
    elif split == "test_prediction":
        transform = no_transform
        glob_d = None
        glob_gt = None  #"test_depth_completion_anonymous/"
        glob_rgb = os.path.join(
            args.data_folder,
            "kitti_dc_4k/depth_selection/test_depth_prediction_anonymous/image/*.png")
    else:
        raise ValueError("Unrecognized split " + str(split))

    if glob_gt is not None:
        # train or val-full or val-select
        paths_d = sorted(glob.glob(glob_d)) 
        paths_gt = sorted(glob.glob(glob_gt)) 
        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
        #print("paths_rgb:{}".format(paths_rgb))
    else:  
        # test only has d or rgb
        paths_rgb = sorted(glob.glob(glob_rgb))
        paths_gt = [None] * len(paths_rgb)
        if split == "test_prediction":
            paths_d = [None] * len(
                paths_rgb)  # test_prediction has no sparse depth
        else:
            paths_d = sorted(glob.glob(glob_d))

    if len(paths_d) == 0 and len(paths_rgb) == 0 and len(paths_gt) == 0:
        raise (RuntimeError("Found 0 images under {}".format(glob_gt)))
    if len(paths_d) == 0 and args.use_d:
        raise (RuntimeError("Requested sparse depth but none was found"))
    if len(paths_rgb) == 0 and args.use_rgb:
        raise (RuntimeError("Requested rgb images but none was found"))
    if len(paths_rgb) == 0 and args.use_g:
        raise (RuntimeError("Requested gray images but no rgb was found"))
    if len(paths_rgb) != len(paths_d) or len(paths_rgb) != len(paths_gt):
        raise (RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    # print(len(paths["rgb"]))
    return paths, transform
```



# Train
```bash
$ python FCFRNet_Paddle/main.py --train-mode dense -b 1 --input rgbd
```

# Test
```bash
$python main.py --evaluate bestmodel.pdparams --val select --input rgbd
```
We Use Torch2Paddle to get .pdparams weight to valuate in kitti. 
| Data                                     | RMSE   | MAE    | iRMSE | RMSE1 | RMSE2 | iMAE |
| ---------------------------------------- | ------ | ------ | --------- | --------- | --------- | --------- |
| `Kitti(Fast Unpool, pos affinity)`  Paddle |784.224 | 222.639| 2.370    | 784.224   | 784.224 |1.014|


