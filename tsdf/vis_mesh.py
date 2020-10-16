import open3d as o3d

# fname = "single.ply"
fname = "mesh.ply"

mesh = o3d.io.read_triangle_mesh(fname)
o3d.visualization.draw_geometries([mesh])