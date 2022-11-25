## FCFRNet/dataloader/kitti_loader.py

for example,when you are training,you need set input type as rgbd.That is say  you need kitti raw groundtruth,raw rgb image,and dense depth input.Need to Attention is that dense depth images  are the result of frist stage(sparse to coarse step).We use STD to get them.You can change it as any other framework.so you need to change the following glob_d(dense depth input) glob_gt(groundtruth) get_rgb_path(rgb image path)

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

