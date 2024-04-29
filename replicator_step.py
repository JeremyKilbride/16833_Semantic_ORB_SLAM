import argparse
from omni.isaac.kit import SimulationApp
import time

start=time.time()
#basic configuration settings
config = {
    "launch_config": {
        "renderer": "RayTracedLighting",
        "headless": True,},
    "writer": "BasicWriter",
    "writer_config": {
        "output_dir": "/home/jkilbrid/_out_offline_generation/warehouse_two_forklifts_close",
        "rgb": True,
        "bounding_box_2d_tight": True,
        "semantic_segmentation": True,
        "instance_segmentation" : True,
        "distance_to_image_plane": False,
        "bounding_box_3d": False,
        "occlusion": False,},
    "num_frames":10,
    "rt_subframes":4,
    "resolution": [2048, 2048],
    "asset_path":"omniverse://cerlabnucleus.lan.local.cmu.edu/",
    "env_path":"omniverse://cerlabnucleus.lan.local.cmu.edu/Library/Asset_Pack_1/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "rotation": 360
}

#handle input arguements
parser=argparse.ArgumentParser()
parser.add_argument("-n","--num_frames",type=int)
args=parser.parse_args()
if args.num_frames:
    print(f"got number of frames: {args.num_frames}")
    config["num_frames"]=args.num_frames
else:
    print(f"number of frames not specified, using default value of {config['num_frames']}")

#start the simulation app
simapp=SimulationApp(launch_config=config["launch_config"])

import omni.replicator.core as rep
from omni import usd
from  omni.isaac.core.world import World
from omni.isaac.core.objects import  VisualCone, VisualCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
import omni.isaac.core.utils.prims as prims
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
import omni.isaac.core.utils.rotations as rotations
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import get_current_stage, open_stage, print_stage_prim_paths
import omni.isaac.core.utils.physics as physics_utils
from omni.isaac.sensor import Camera
from pxr import Gf, UsdGeom, UsdGeom, UsdShade, Sdf



import numpy as np

#get paths for usd files
asset_path=config['asset_path']
husky_path=asset_path +'Users/gmetts/theia_isaac_qual/robots/theia_robot/theia_robot.usd'
forklift_path=asset_path +"Users/gmetts/theia_isaac_qual/robots/forklift/forklift.usd"
wheel_loader_path=asset_path+'Users/gmetts/wheel_loader/WheelLoader.usdc'
forklift2_path=asset_path+"Library/Asset_Pack_1/Isaac/Robots/Forklift/forklift_b.usd"

#setup the scene
if open_stage(config["env_path"]):
    print('stage opened successfully')
    stage=get_current_stage()
else:
    print('unable to get stage, exting the application')
    simapp.close()	

remove_all_semantics(prims.get_prim_at_path('/Root'),recursive=True) #remove semantics from the environment

#add husky to stage and add the semantics
add_reference_to_stage(forklift2_path,'/World/forklift2')
add_update_semantics(prims.get_prim_at_path('/World/forklift2'),"forklift",type_label='class')
# add_reference_to_stage(wheel_loader_path,'/World/Wheel_loader')
# add_update_semantics(prims.get_prim_at_path('/World/Wheel_loader'),"wheel_loader",type_label='class')
add_update_semantics(prims.get_prim_at_path('/Root/forklift'),"forklift",type_label='class')
world=World()
world.get_physics_context().set_gravity(0.0)
# wheel_prims=prims.find_matching_prim_paths("/World/Wheel_loader/Loader_*/*")		#find prim paths for the wheel loader
lift2_prims=prims.find_matching_prim_paths("/World/forklift2/*")		#find prim paths for the wheel loader
lift2=world.scene.add(Articulation("/World/forklift2", name="forklift2",position=(1.5,-0.25,0),orientation=rotations.euler_angles_to_quat((0,0,90),degrees=True)))
# wheel1=world.scene.add(Articulation("/World/Wheel_loader", name="Wheel",position=(0,0,0),scale=(0.01,)))
lift1=Articulation('/Root/forklift', name='forklift1')
lift1.set_world_pose(position=np.array([1.5,3.5,0]))




def remove_rigid_boy(prims_list):
    for prim in prims_list:									#search prims at each prim path and disable the rigid body api if it is enabled
        if physics_utils.get_rigid_body_enabled(prim):
            physics_utils.set_rigid_body_enabled(False, prim)
            print(f"rigid body disabled for {prim}")

remove_rigid_boy(lift2_prims)

def rotate(self, prim, rotation):      
    # quat = rotations.euler_angles_to_quat(np.array(rotation))
    quat = Gf.Quatf(rotation[0], tuple(rotation[1:]))
    prim.GetAttribute("xformOp:orient").Set(quat)


def look_at(self, camera_prim, target_location):
    world_transformation_matrix = usd.get_world_transform_matrix(camera_prim) # matrix from camera to world

    camera_location = world_transformation_matrix.ExtractTranslation() # location of the camera
    direction = target_location - camera_location
    z_direction = -np.array(direction / np.linalg.norm(direction))  # the local z axis of camera points backwards so we take negative of direction
    
    vec = np.array([0, 0, 1]) # a vector in global up direction
    if np.allclose(z_direction, vec):
        vec = np.array([1, 0, 0]) # choose a different vector not parallel to z_direction
    
    # Get the local x and y directions of the camera using cross product
    x_direction = np.cross(vec, z_direction) # x is a vector perpendicular to z and global up
    x_direction = x_direction / np.linalg.norm(x_direction)
    
    y_direction = np.cross(z_direction, x_direction) 
    y_direction = y_direction / np.linalg.norm(y_direction)

    x_direction = x_direction.reshape(3, 1)
    y_direction = y_direction.reshape(3, 1)
    z_direction = z_direction.reshape(3, 1)

    rotation_matrix = np.concatenate([x_direction, y_direction, z_direction], axis=1)
    
    # angles = rotations.matrix_to_euler_angles(rotation_matrix)
    # angles = np.rad2deg(angles)
    quat = rotations.rot_matrix_to_quat(rotation_matrix)
    self.rotate(camera_prim, quat)

def increment_roation(angle,wheel):
    _, current_orientation=wheel.get_world_pose()
    print(f'current orientation: {rotations.quat_to_euler_angles(current_orientation,degrees=True)}')
    wheel.set_world_pose(orientation=rotations.euler_angles_to_quat((0,0,angle), degrees=True))
    return None

def randomize_vehicles(husky,wheel,lift):
    area1_low,area1_up=(-10,-10,0),(3,-5,0)
    area2_low,area2_up=(-5,-4,0),(3,3,0)
    area3_low,area3_up=(-10,-4,0),(-5,3,0)
    orders=[123,213,321]
    order=np.random.choice(np.array(orders))
    # print(f"got order: {order}")
    match order:
        case 123:
            area1_low,area1_up=(-10,-10,0.15),(3,-5,0.15)
            husky.set_world_pose(position=np.random.uniform(area1_low,area1_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))
            wheel.set_world_pose(position=np.random.uniform(area2_low,area2_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))
            lift.set_world_pose(position=np.random.uniform(area3_low,area3_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))
        case 321:
            area3_low,area3_up=((-10,-4,0.15),(-5,3,0.15))
            husky.set_world_pose(position=np.random.uniform(area3_low,area3_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))
            wheel.set_world_pose(position=np.random.uniform(area2_low,area2_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))
            lift.set_world_pose(position=np.random.uniform(area1_low,area1_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))
        case 213:
            area2_low,area2_up=((-5,-4,0.15),(3,3,0.15))
            husky.set_world_pose(position=np.random.uniform(area2_low,area2_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))
            wheel.set_world_pose(position=np.random.uniform(area1_low,area1_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))
            lift.set_world_pose(position=np.random.uniform(area3_low,area3_up), orientation=rotations.euler_angles_to_quat(np.random.uniform((0,0,-120),(0,0,120)), degrees=True))

for i in range(10):
    simapp.update()

camera = world.scene.add(Camera("/World/Camera", name="Camera1", position=(-16.5,-0.75,2), orientation=rotations.euler_angles_to_quat((0,0,0),degrees=True)))
render_product = rep.create.render_product("/World/Camera", config["resolution"])

writer = rep.WriterRegistry.get(config["writer"])

writer.initialize(**config["writer_config"])

writer.attach([render_product])
increment=config["rotation"]/config["num_frames"]
angles=np.arange(start=0,step=increment,stop=config["rotation"])
print(angles)
print(f'angle increment is {increment}')
for k in range(config["num_frames"]):
    increment_roation(angles[k],lift1)
    increment_roation(angles[k],lift2)
    rep.orchestrator.step(rt_subframes=config['rt_subframes'])
    # simapp.update()
# while True:
#     simapp.update()
end = time.time()
total_time=end-start
print(f'total time:{total_time} seconds')
simapp.close()
