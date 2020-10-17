import open3d as o3d

# fname = "single.ply"
# fname = "single_ours.ply"
# fname = "single_gt.ply"
# fname = "mesh.ply"
# fname = "mesh_ours.ply"
# fname = "mesh_gt.ply"
single = -1
mesh = -1
more = -1

fname = "mesh.ply"

single = o3d.io.read_triangle_mesh("single.ply")
# mesh = o3d.io.read_triangle_mesh("mesh.ply")
# more = o3d.io.read_triangle_mesh(fname)

tmp = [single, mesh, more]

geos = []
for t in tmp:
    if t != -1:
        geos.append(t)


o3d.visualization.draw_geometries(geos)