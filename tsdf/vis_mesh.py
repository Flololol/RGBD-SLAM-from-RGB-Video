import open3d as o3d

# fname = "single.ply"
fname = "single_ours.ply"
# fname = "single_gt.ply"
# fname = "mesh.ply"
# fname = "mesh_ours.ply"
# fname = "mesh_gt.ply"

mesh = o3d.io.read_triangle_mesh(fname)
o3d.visualization.draw_geometries([mesh])