You can download the dataset from [here](https://github.com/aalto-intelligent-robotics/shape_completion_network#download-and-parse-training-and-test-data)

The folder structure is the same, but you will need to change the format of the partial point clouds to `npy`.
**pcd.npy** containing the 3D list of points (N, 3), **vertices.npy** and
**triangles.npy** containing the triangles and verticies of the mesh.

This is simple enough:
- 1 - Read the files with `open3d` or a library of your choice
- 2 - Access the triangles and vertices or the partial point cloud and convert them to a numpy array
- 3 - Dump them into an `.npy` file

Unfortunately I can't find the original script but if you need help with it open an issue.

Alternatively you can change the code in `ShapeCompletionDataset.py` to directly read the data in the original format.
I moved to `.npy` because of `open3d` having problem loading the files concurrently but they could have fixed it now.
