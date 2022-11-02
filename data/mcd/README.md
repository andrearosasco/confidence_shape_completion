You can download the dataset from [here](https://github.com/aalto-intelligent-robotics/shape_completion_network#download-and-parse-training-and-test-data)

The folder structure is the same, but you will need to change to convert the partial point clouds to npy files
(**pcd.npy**) containing the 3D list of points (N, 3) and the meshes to 2 npy files: **vertices.npy** (N, 3) and
**triangles.npy** (N, 3).

This is simple enough:
- 1 - Read the files with `open3d` or a library of your choice
- 2 - Access the triangles and vertices or the partial point cloud and convert them to a numpy array
- 3 - Dump them into an `.npy` file

Unfortunately I can't find the original script but if you need to do that and can't do it on your own open an issue,
and I'll write a new one.

Alternatively you can change the code in `ShapeCompletionDataset.py` to directly read the data in the original format.
I moved to `.npy` because of `open3d` having problem loading the files concurrently but they could have fixed it now.