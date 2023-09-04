import laspy
import nksr
import numpy as np
import pyproj
import torch
from pycg import vis

if __name__ == "__main__":
    las_data = laspy.read("assets/las/example.las")

    # Determine coordinate system if available
    crs = None
    for vlr in las_data.header.vlrs:
        if isinstance(vlr, laspy.vlrs.known.WktCoordinateSystemVlr):
            crs = pyproj.CRS.from_wkt(vlr.string)
            print("Found coordinate system", crs)
            break

    # Dump xyz coordinates
    xyz = np.vstack((las_data.x, las_data.y, las_data.z)).T
    print("Point stats (mean, min, max)", np.mean(xyz, axis=0), np.min(xyz, axis=0), np.max(xyz, axis=0))

    xyz_offset = (np.max(xyz, axis=0) + np.min(xyz, axis=0)) / 2.0
    xyz -= xyz_offset[None, :]

    # Crop a small region of interest
    xyz = xyz[np.linalg.norm(xyz, axis=1) < 20.0]
    vis.show_3d([vis.pointcloud(xyz)])

    # Apply NKSR
    device = torch.device("cuda:0")
    reconstructor = nksr.Reconstructor(device)
    reconstructor.chunk_tmp_device = torch.device("cpu")

    input_xyz = torch.from_numpy(xyz).float().to(device)
    input_sensor = torch.tensor([[0.0, 0.0, 50.0]], device=device).repeat(input_xyz.shape[0], 1)

    field = reconstructor.reconstruct(
        input_xyz, sensor=input_sensor, detail_level=0.1,
        approx_kernel_grad=True, solver_tol=1e-4, fused_mode=True, 
        preprocess_fn=nksr.get_estimate_normal_preprocess_fn(64, 85.0)
    )

    mesh = field.extract_dual_mesh(mise_iter=1)

    # Convert the mesh to WGS-84 coordinate system if possible
    mesh.v = mesh.v.cpu().numpy().astype(float)
    mesh.v += xyz_offset[None, :]
    if crs is not None:
        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", allow_ballpark=False)
        res = transformer.transform(xx=mesh.v[:, 0], yy=mesh.v[:, 1], zz=mesh.v[:, 2])
        mesh.v = np.stack(res, axis=1)

    mesh = vis.mesh(mesh.v, mesh.f)
    vis.to_file(mesh, "assets/las/example.obj")
