# Lib import
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation

# Get model
feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")

# Load and resize image
image = Image.open(r"path\scene.png").convert("RGB")
# plt.imshow(image)
new_height = 480 if image.height > 480 else image.height
new_height -= (new_height % 32)
new_width = int(new_height * image.width / image.height)
diff = new_width % 32
# plt.show()

new_width = new_width - diff if diff < 16 else new_width + 32 - diff
new_size = (new_width, new_height)
image = image.resize(new_size)
# plt.imshow(image)
# plt.show()

# Prepare image for model

inputs = feature_extractor(images= image,  return_tensors= "pt")

# Getting pred
with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth= outputs.predicted_depth

# Post-processing
pad = 16 
output = predicted_depth.squeeze().cpu().numpy() * 1000.0 
output = output[pad:-pad, pad:-pad] 
image = image.crop((pad, pad, image.width - pad, image.height - pad)) 
# visualize the prediction 
fig, ax = plt.subplots(1, 2) 
ax[0].imshow(image) 
ax[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
ax[1].imshow(output, cmap='plasma') 
ax[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False) 
plt.tight_layout() 
plt.pause(5) 

# Importing libs
import numpy as np
import open3d as o3d

# preapre the depth image for open3d

width, height = image.size 

depth_image = (output * 255 / np.max(output)).astype('uint8') 
image = np.array(image) 

# create rgbd image 
depth_o3d = o3d.geometry.Image(depth_image) 
image_o3d = o3d.geometry.Image(image) 
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d, depth_o3d, convert_rgb_to_intensity=False)

# Creating a Camera 
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic() 
camera_intrinsic.set_intrinsics(width, height, 500, 500, width/2, height/2) 

# Creating o3d point cloud
pcd_raw = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)

o3d.visualization.draw_geometries([pcd_raw])

# Post-processing 3D point cloud

# outliers removal 
cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=20, std_ratio=6.0) 
pcd = pcd_raw.select_by_index(ind)

# estimate normals
pcd.estimate_normals() 
pcd.orient_normals_to_align_with_direction() 

o3d.visualization.draw_geometries([pcd]) 

# Surface reconstruction 
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10, n_threads=1)[0] 

# rotate the mesh 
rotation = mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0)) 
mesh.rotate(rotation, center=(0, 0, 0)) 

# visualize the mesh 
o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True) 

# mesh_uniform = mesh.paint_uniform_color([0.9, 0.8, 0.9]) 
# mesh_uniform.compute_vertex_normals() 
# o3d.visualization.draw_geometries([mesh_uniform], mesh_show_back_face=True) 

# 3D Mesh Export 
o3d.io.write_triangle_mesh(r'path\RESULTS\off.obj', mesh)
o3d.io.write_triangle_mesh(r'path\RESULTS\on.ply', mesh) 