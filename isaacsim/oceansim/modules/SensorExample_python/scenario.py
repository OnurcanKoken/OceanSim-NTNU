# Omniverse import
import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, Usd, UsdPhysics
import time
import rclpy

from geometry_msgs.msg import Quaternion, Vector3, Pose, PoseStamped, TransformStamped, Wrench
from nav_msgs.msg import Path
from tf2_ros import TransformBroadcaster

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
import omni.timeline

from isaacsim.asset.gen.omap.bindings import _omap

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.util.debug_draw")
from isaacsim.util.debug_draw import _debug_draw

from isaacsim.oceansim.sensors.datacollection import DataCollectionSensor
from isaacsim.oceansim.utils.occupancy_map import OccupancyMap, Point2d
from isaacsim.oceansim.utils.path_planner import generate_random_path
from isaacsim.oceansim.utils.assets_utils import get_map_config_path, get_data_collection_root
import os
import PIL.ImageDraw
import time
import csv

from rosgraph_msgs.msg import Clock

# ROS Control import
try:
    from isaacsim.oceansim.utils.ros2_control import ROS2ControlReceiver
    ROS2_CONTROL_AVAILABLE = True
    print("[Scenario] Simple ROS2 Control receiver found")
except ImportError as e:
    ROS2_CONTROL_AVAILABLE = False
    print(f"[Scenario] Simple ROS2 Control not available: {e}")
    print("[Scenario] ROS2 Control functionality will be disabled")

class MHL_Sensor_Example_Scenario():
    def __init__(self, publish_clock=True, publish_pose=True, publish_map=False, publish_cmds=True):
        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._IMU = None
        self._data_collection_mode = False
        self.data_collection_path = ""
        self.waypoints_control_speed = True


        self._ctrl_mode = None
        self.waypoints = []

        self._running_scenario = False
        self._time = 0.0

        # ROS2 Control
        self._ros2_control_receiver = None
        self._enable_ros2_control = True
        self._ros2_control_mode = "velocity control"

        self._rob_pose_topic = "/oceansim/robot/pose"
        self._rob_cmd_topic = "/oceansim/robot/cmd"
        self._publish_clock = publish_clock
        self._publish_pose = publish_pose
        self._publish_map = publish_map
        self._publish_cmds = publish_cmds

        # Initialize ROS2 context if not already done
        if not rclpy.ok():
            rclpy.init()
            print('ROS2 context initialized')


        if self._publish_cmds or self._publish_pose:
            self._init_ros2_publishers()

        self._timeline = omni.timeline.get_timeline_interface()

    def _init_ros2_publishers(self):
        """Initialize ROS2 publishers and subscribers."""
        if self._publish_cmds:
            # Create commands publisher node
            node_name = f'oceansim_rob_cmd_pub'
            self._ros2_rob_cmd_node = rclpy.create_node(node_name)
            self._rob_cmd_pub = self._ros2_rob_cmd_node.create_publisher(
                Wrench,
                self._rob_cmd_topic,
                10
            )

        if self._publish_clock:
            # Create clock publisher node
            node_name = f'oceansim_clock_pub'
            self._ros2_clock_node = rclpy.create_node(node_name)
            self._clock_pub = self._ros2_clock_node.create_publisher(
                Clock,
                '/clock',
                10
            )

        self._rob_pose_pub = None
        if self._publish_pose:
            # Create pose publisher node
            node_name = f'oceansim_rob_pose_pub'
            self._ros2_rob_pose_node = rclpy.create_node(node_name)
            self._rob_pose_pub = self._ros2_rob_pose_node.create_publisher(
                PoseStamped,
                self._rob_pose_topic,
                10
            )

            # Path Publisher
            self._path_topic = "/oceansim/robot/path"
            self._path_pub = self._ros2_rob_pose_node.create_publisher(
                Path,
                self._path_topic,
                10
            )
            self._path_msg = Path()
            self._path_msg.header.frame_id = 'map' # Path is in map frame

            self._last_path_pos = None
            self._path_update_threshold = 0.05  # Only add point if moved self._path_update_threshold meters

            # TF Broadcaster
            self._tf_broadcaster = TransformBroadcaster(self._ros2_rob_pose_node)

            if self._publish_map:
                self._omap_generator = None
                self._last_omap_pos = None
                self._omap_update_threshold = 0.5
                
                # Initialize Debug Draw interface (Safe to do here)
                if _debug_draw is not None:
                    self._debug_draw = _debug_draw.acquire_debug_draw_interface()
                else:
                    self._debug_draw = None

    def _init_manual_controllers(self):
        """Initialize manual control interfaces (Keyboard/Gamepad)."""
        from ...utils.keyboard_cmd import keyboard_cmd
        from ...utils.gamepad_cmd import gamepad_cmd
        import carb.input

        self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)
        self._force_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                  input_keyboard_mapping={
                                    # forward command
                                    "W": [10.0, 0.0, 0.0],
                                    # backward command
                                    "S": [-10.0, 0.0, 0.0],
                                    # leftward command
                                    "A": [0.0, 10.0, 0.0],
                                    # rightward command
                                    "D": [0.0, -10.0, 0.0],
                                     # rise command
                                    "UP": [0.0, 0.0, 10.0],
                                    # sink command
                                    "DOWN": [0.0, 0.0, -10.0],
                                  })
        self._torque_cmd = keyboard_cmd(base_command=np.array([0.0, 0.0, 0.0]),
                                  input_keyboard_mapping={
                                    # yaw command (left)
                                    "J": [0.0, 0.0, 10.0],
                                    # yaw command (right)
                                    "L": [0.0, 0.0, -10.0],
                                    # pitch command (up)
                                    "I": [0.0, -10.0, 0.0],
                                    # pitch command (down)
                                    "K": [0.0, 10.0, 0.0],
                                    # row command (left)
                                    "LEFT": [-10.0, 0.0, 0.0],
                                    # row command (negative)
                                    "RIGHT": [10.0, 0.0, 0.0],
                                  })
        self._joy_force = gamepad_cmd(
            input_mapping={
                carb.input.GamepadInput.LEFT_STICK_UP:    np.array([1.0, 0.0, 0.0]),   # Forward
                carb.input.GamepadInput.LEFT_STICK_DOWN:  np.array([-1.0, 0.0, 0.0]),  # Backward
                carb.input.GamepadInput.LEFT_STICK_LEFT:  np.array([0.0, 1.0, 0.0]),   # Left
                carb.input.GamepadInput.LEFT_STICK_RIGHT: np.array([0.0, -1.0, 0.0]),  # Right
                carb.input.GamepadInput.RIGHT_TRIGGER:    np.array([0.0, 0.0, 1.0]),   # Up
                carb.input.GamepadInput.LEFT_TRIGGER:     np.array([0.0, 0.0, -1.0]),  # Down
            },
            scale=9.0 
        )
        
        self._joy_torque = gamepad_cmd(
            input_mapping={
                # Pitch
                carb.input.GamepadInput.RIGHT_STICK_UP:    np.array([0.0, -1.0, 0.0]), 
                carb.input.GamepadInput.RIGHT_STICK_DOWN:  np.array([0.0, 1.0, 0.0]),
                # Yaw
                carb.input.GamepadInput.RIGHT_STICK_LEFT:  np.array([0.0, 0.0, 1.0]),
                carb.input.GamepadInput.RIGHT_STICK_RIGHT: np.array([0.0, 0.0, -1.0]),
                # Roll
                carb.input.GamepadInput.LEFT_SHOULDER:     np.array([-1.0, 0.0, 0.0]),
                carb.input.GamepadInput.RIGHT_SHOULDER:    np.array([1.0, 0.0, 0.0]),
            },
            scale=4.0
        )

    def _setup_data_logging_for_sensors(self, uw_yaml_path):
        """Setup data collection and logging for all sensors."""
        self.setup_data_collection(data_path=self.data_collection_path)
        if self._sonar is not None:
            # sensor_name = "sonar_sensor"
            # sensor_path = self._data_collector.collect_data(name=sensor_name)
            self._sonar.sonar_initialize(include_unlabelled=True)
        if self._cam is not None:
            sensor_name = "camera_sensor"
            sensor_path = self._data_collector.collect_data(name=sensor_name)
            self._cam.initialize(writing_dir=sensor_path, ros2_pub_frequency=self._cam.get_frequency(), UW_yaml_path=uw_yaml_path)
        if self._DVL is not None:
            sensor_name = "DVL_sensor"
            sensor_path = self._data_collector.collect_data(name=sensor_name)
            self._DVL_reading = [0.0, 0.0, 0.0]
            self._DVL.init_logging(sensor_path)

        if self._baro is not None:
            sensor_name = "barometer_sensor"
            sensor_path = self._data_collector.collect_data(name=sensor_name)
            self._baro_reading = 101325.0 # atmospheric pressure (Pa)
            self._baro.init_logging(sensor_path)

        # Ground Truth (Trajectory)
        sensor_name = "ground_truth"
        sensor_path = self._data_collector.collect_data(name=sensor_name)

        self._gt_csv_file = open(os.path.join(sensor_path, "trajectory.csv"), 'w', newline='')
        self._gt_csv_writer = csv.writer(self._gt_csv_file)
        self._gt_csv_writer.writerow(['timestamp', 'p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z'])
        if self._IMU is not None:
            sensor_name = "IMU_sensor"
            sensor_path = self._data_collector.collect_data(name=sensor_name)
            self._IMU.initialize()
            self._IMU.save_metadata(save_path=sensor_path)
            self._IMU.init_logging(save_path=sensor_path) # <--- Init CSV Logging
            self._IMU_reading = {
                'linear_acceleration': np.array([0.0, 0.0, 0.0]),
                'angular_velocity':    np.array([0.0, 0.0, 0.0]),
                'orientation':         np.array([1.0, 0.0, 0.0, 0.0]),
                'time':                0.0,
                'physics_step':        0    
            }

    def setup_scenario(self, rob, sonar, cam, DVL, baro, IMU, ctrl_mode,data_collection_mode, data_collection_path="", uw_yaml_path=None):
        if not rclpy.ok():
            print("[Scenario] ROS2 Context was dead. Resurrecting before sensor init...")
            rclpy.init()

        self._data_collection_mode = data_collection_mode
        self.data_collection_path = data_collection_path
        self._rob = rob
        self._sonar = sonar
        self._cam = cam
        self._DVL = DVL
        self._baro = baro
        self._IMU = IMU
        self._ctrl_mode = ctrl_mode
        
        # Initialize ROS2 publishers for DVL and Barometer
        if self._DVL is not None:
            self._DVL.initialize_ros2()
        
        if self._baro is not None:
            self._baro.initialize_ros2()

        # Data collection setup
        if self._data_collection_mode:
            self._setup_data_logging_for_sensors(uw_yaml_path)
        else:
            if self._sonar is not None:
                self._sonar.sonar_initialize(include_unlabelled=True)
            if self._cam is not None:
                self._cam.initialize(ros2_pub_frequency=self._cam.get_frequency(), UW_yaml_path=uw_yaml_path)#, writing_dir="/home/osim-mir/OceanSimAssets/GroundTruth")
            if self._DVL is not None:
                self._DVL_reading = [0.0, 0.0, 0.0]
            if self._baro is not None:
                self._baro_reading = 101325.0 # atmospheric pressure (Pa)
            if self._IMU is not None:
                self._IMU.initialize()
                self._IMU_reading = {
                    'linear_acceleration': np.array([0.0, 0.0, 0.0]),
                    'angular_velocity':    np.array([0.0, 0.0, 0.0]),
                    'orientation':         np.array([1.0, 0.0, 0.0, 0.0]),
                    'time':                0.0,
                    'physics_step':        0    
                }

        try:
            self._physx_interface = omni.physx.acquire_physx_interface()
            self._stage_id = omni.usd.get_context().get_stage_id()
            
            # Check if PhysicsScene exists (Optional safety check)
            stage = omni.usd.get_context().get_stage()
            has_physics_scene = False
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Scene):
                    has_physics_scene = True
                    break
            
            if has_physics_scene:
                self._omap_generator = _omap.Generator(self._physx_interface, self._stage_id)
                self._omap_generator.update_settings(0.2, 4, 5, 6)
                print("[Scenario] Occupancy Map Generator Initialized.")
            else:
                print("[Scenario] WARNING: No PhysicsScene found on stage. OMap disabled.")
        except Exception as e:
            print(f"[Scenario] Failed to init OMap Generator: {e}")
            self._omap_generator = None

        # Apply the physx force schema if manual control
        if ctrl_mode == "Manual control" or ctrl_mode == "ROS + Manual control":
            self._init_manual_controllers()

        if ctrl_mode == "ROS control" or ctrl_mode == "ROS + Manual control":
            self._rob_forceAPI = PhysxSchema.PhysxForceAPI.Apply(self._rob)

            # initialize ROS2ControlReceiver
            self._setup_ros2_control()
            
        self._running_scenario = True

    def _setup_ros2_control(self):
        """setup ROS2 control receiver"""
        if not ROS2_CONTROL_AVAILABLE:
            return
        
        try:
            self._ros2_control_receiver = ROS2ControlReceiver(self._rob, "ROS2ControlReceiver")
            
            if hasattr(self, '_rob_forceAPI') and self._rob_forceAPI is not None:
                self._ros2_control_receiver.set_scenario_force_api(self._rob_forceAPI)

            self._ros2_control_receiver.initialize(
                enable_ros2=True
            )

            self._ros2_control_receiver._setup_ros2_control_mode(
                self._ros2_control_mode
            )
                
        except Exception as e:
            print(f"[Scenario] setup ros2 control receiver failed: {e}")
            self._ros2_control_receiver = None

    # This function will only be called if ctrl_mode==waypoints and waypoints files are changed
    def generate_random_waypoints(self):
        print("[Scenario] Generating random waypoints...")
        # Resolve path to map.yaml via config
        map_yaml_path = get_map_config_path()
        map_yaml_path = os.path.normpath(map_yaml_path)
        
        # Get data collection path/occupancy_map_folder
        data_collection_path = get_data_collection_root()
        data_collection_path = os.path.normpath(data_collection_path)
        save_path = os.path.join(data_collection_path, "occupancy_map")
        if not os.path.exists(save_path):
            os.makedirs(save_path)      
        
        if not os.path.exists(map_yaml_path):
            print(f"[Scenario] Error: map.yaml not found at {map_yaml_path}")
            return

        try:
            omap = OccupancyMap.from_ros_yaml(map_yaml_path)
            
            # Get current robot pose
            if self._rob is not None:
                # self._rob is maybe a Prim or Rig? 
                # Needs logic to get translation. 
                # Assuming UsdGeom.Xformable get translation
                 curr_transform = UsdGeom.Xformable(self._rob).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                 trans = curr_transform.ExtractTranslation()
                 start_pose = Point2d(x=trans[0], y=trans[1])
            else:
                 start_pose = None

            path = generate_random_path(omap, start_pose)
            
            if path is not None:
                # Retrieve current Z and Orientation to maintain stable height/rotation
                # Default values if robot not found
                current_z = -0.8
                current_quat = [1.0, 0.0, 0.0, 0.0] # w, x, y, z

                if self._rob is not None:
                     if self._rob.IsValid():
                         curr_transform = UsdGeom.Xformable(self._rob).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                         trans = curr_transform.ExtractTranslation()
                         rot = curr_transform.ExtractRotationQuat()
                         current_z = trans[2]
                         current_quat = [rot.GetReal(), rot.GetImaginary()[0], rot.GetImaginary()[1], rot.GetImaginary()[2]]
                     else:
                         print("[Scenario] Warning: Robot prim is invalid. Using default start pose Z and orientation.")

                # Convert to list and append Z and Orientation
                # Expectation from update_scenario:
                # self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                # self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                # So format is: [x, y, z, w, x, y, z]
                
                full_waypoints = []
                for p in path:
                    # p is [x, y]
                    point_data = [
                        float(p[0]), 
                        float(p[1]), 
                        float(current_z),
                        float(current_quat[0]),
                        float(current_quat[1]), 
                        float(current_quat[2]), 
                        float(current_quat[3])
                    ]
                    full_waypoints.append(point_data)

                self.waypoints = full_waypoints
                
                self.waypoints = full_waypoints
                
                print(f"[Scenario] Successfully set {len(self.waypoints)} random waypoints (with Z={current_z:.2f}).")
                if len(self.waypoints) > 0:
                     print(f"Waypoint[0]: {self.waypoints[0]}")
                     
                     # ---------------------------------------------------------
                     # Save Map with Path
                     # ---------------------------------------------------------
                     try:
                         # 1. Get Image
                         img = omap.ros_image().convert("RGB") # Convert to RGB to draw colored lines
                         draw = PIL.ImageDraw.Draw(img)
                         
                         # 2. Convert path to pixels
                         # path is numpy array of [x, y]
                         pixels = omap.world_to_pixel_numpy(path)
                         
                         # 3. Draw Lines
                         # pixels is [[x, y], [x, y], ...]
                         pixel_tuples = [tuple(p) for p in pixels]
                         if len(pixel_tuples) > 1:
                            draw.line(pixel_tuples, fill="red", width=2)
                            
                         # Draw Start (Green) and End (Blue) circles
                         start_px = pixel_tuples[0]
                         end_px = pixel_tuples[-1]
                         r = 3
                         draw.ellipse((start_px[0]-r, start_px[1]-r, start_px[0]+r, start_px[1]+r), fill="green")
                         draw.ellipse((end_px[0]-r, end_px[1]-r, end_px[0]+r, end_px[1]+r), fill="blue")

                         # 4. Save
                         timestamp = self._time
                         save_filename = f"generated_path_{timestamp}.png"
                         # save in data_collection_path/occupancy_map
                         save_dir = os.path.join(save_path, save_filename)
                         img.save(save_dir)
                         print(f"[Scenario] Saved generated path image to: {save_dir}")
                         
                     except Exception as e:
                         print(f"[Scenario] Warning: Failed to save path image: {e}")
            else:
                 print("[Scenario] Failed to generate path, falling back to empty.")
                 self.waypoints = []
                 
        except Exception as e:
            print(f"[Scenario] Critical error generating path: {e}")
            import traceback
            traceback.print_exc()

    def setup_waypoints(self, waypoint_path, default_waypoint_path):
        
        if waypoint_path == "RANDOM":
             self.generate_random_waypoints()
             return

        try:
            self.waypoints = self._read_waypoints_from_file(waypoint_path)
            print('Waypoints loaded successfully.')
            print(f'Waypoint[0]: {self.waypoints[0]}')
        except:
            self.waypoints = self._read_waypoints_from_file(default_waypoint_path)
            print('Fail to load this waypoints. Back to default waypoints.')

    def _read_waypoints_from_file(self, file_path):
        # Initialize an empty list to store the floats
        data = []
        
        # Open the file in read mode
        with open(file_path, 'r') as file:
            # Read each line in the file
            for line in file:
                # Strip any leading/trailing whitespace and split the line by spaces
                float_strings = line.strip().split()
                
                # Convert the list of strings to a list of floats
                floats = [float(x) for x in float_strings]
                
                # Append the list of floats to the data list
                data.append(floats)
        
        return data

    def setup_data_collection(self, data_path):
        if data_path is None or data_path=="":
            data_path = get_data_collection_root()
        else:
            data_path = data_path
        self.data_collection_path = data_path
        try:
            self._data_collector = DataCollectionSensor(data_path=self.data_collection_path)
        except Exception as e:
            print(f"[Scenario] Error initializing DataCollectionSensor: {e}")
            import traceback
            traceback.print_exc()

        
    def teardown_scenario(self):

        # Because these two sensors create annotator cache in GPU,
        # close() will detach annotator from render product and clear the cache.
        if self._sonar is not None:
            self._sonar.close()
        if self._cam is not None:
            self._cam.close()
        if self._IMU is not None:
            self._IMU.close()
        
        if hasattr(self, '_gt_csv_file') and self._gt_csv_file:
            try:
                self._gt_csv_file.close()
                self._gt_csv_file = None
                print("Closed Ground Truth log file.")
            except Exception as e:
                print(f"Error closing GT log: {e}")
        
        if self._DVL is not None:
             self._DVL.cleanup()
             
        if self._baro is not None:
             self._baro.cleanup()
        
        # Reset simple variables
        self._time = 0.0

        # clear the keyboard subscription
        if self._ctrl_mode=="Manual control" or self._ctrl_mode=="ROS + Manual control":
            self._force_cmd.cleanup()
            self._torque_cmd.cleanup()
            # self._joy_force.cleanup()
            # self._joy_torque.cleanup()

        # clear the ROS2 control receiver
        if self._ros2_control_receiver is not None:
            self._ros2_control_receiver.close()

        self._rob = None
        self._sonar = None
        self._cam = None
        self._DVL = None
        self._baro = None
        self._IMU = None
        self._running_scenario = False
        self._time = 0.0


    def destroy(self):
        """Cleanup persistent resources including ROS2 nodes"""
        self.teardown_scenario()
        
        # Destroy ROS2 nodes created in __init__
        try:
            if hasattr(self, '_rob_cmd_pub') and self._rob_cmd_pub:
                self._rob_cmd_pub.destroy()
            if hasattr(self, '_ros2_rob_cmd_node') and self._ros2_rob_cmd_node:
                self._ros2_rob_cmd_node.destroy_node()
                print("[Scenario] Destroyed cmd node")
                
            if hasattr(self, '_rob_pose_pub') and self._rob_pose_pub:
                self._rob_pose_pub.destroy()
            if hasattr(self, '_path_pub') and self._path_pub:
                self._path_pub.destroy()
            if hasattr(self, '_ros2_rob_pose_node') and self._ros2_rob_pose_node:
                self._ros2_rob_pose_node.destroy_node()
                print("[Scenario] Destroyed pose node")
                
        except Exception as e:
            print(f"[Scenario] Error destroying ROS nodes: {e}")

    def _publish_robot_pose_and_path(self):
        """Publish robot pose, TF, and update path."""
        # Create a ROS2 Imu message
        msg = PoseStamped()

        sim_time = self._timeline.get_current_time()  # Simulation time
        msg.header.stamp.sec = int(sim_time)
        msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
        msg.header.frame_id = 'map'

        # Get the full 4x4 World Transform Matrix
        world_transform = omni.usd.get_world_transform_matrix(self._rob)

        # Extract Rotation (Quaternion) and Translation
        rot = world_transform.ExtractRotationQuat() 
        trans = world_transform.ExtractTranslation()

        # Populate Message
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])

        msg.pose.orientation.w = float(rot.GetReal())
        msg.pose.orientation.x = float(rot.GetImaginary()[0])
        msg.pose.orientation.y = float(rot.GetImaginary()[1])
        msg.pose.orientation.z = float(rot.GetImaginary()[2])
        
        # Publish the message
        self._rob_pose_pub.publish(msg)

        # Publish TF (Transform from 'map' to 'base_link')
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = 'map'
        tf_msg.child_frame_id = 'base_link' 

        tf_msg.transform.translation.x = msg.pose.position.x
        tf_msg.transform.translation.y = msg.pose.position.y
        tf_msg.transform.translation.z = msg.pose.position.z
        tf_msg.transform.rotation = msg.pose.orientation

        self._tf_broadcaster.sendTransform(tf_msg)

        # Publish Path (Full trajectory)
        current_pos_np = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        update_path = False
        if self._last_path_pos is None:
            update_path = True
        else:
            # Calculate distance moved
            dist = np.linalg.norm(current_pos_np - self._last_path_pos)
            if dist > self._path_update_threshold:
                update_path = True
        
        if update_path:
            self._path_msg.header.stamp = msg.header.stamp
            self._path_msg.poses.append(msg)
            self._path_pub.publish(self._path_msg)
            
            # Update tracker
            self._last_path_pos = current_pos_np

        return current_pos_np, trans

    def _update_occupancy_map(self, current_pos_np, trans):
        if self._debug_draw is not None and self._omap_generator is not None:
            update_omap = False
            
            # Check if this is the first run
            if self._last_omap_pos is None:
                update_omap = True
            else:
                # Check distance from last generation point
                dist_omap = np.linalg.norm(current_pos_np - self._last_omap_pos)
                if dist_omap > self._omap_update_threshold:
                    update_omap = True
            
            if update_omap:
                # 1. Update Transform (Center on Robot)
                self._omap_generator.set_transform(
                    (float(trans[0]), float(trans[1]), float(trans[2])), 
                    (-2.0, -2.0, -2.0), 
                    (2.0, 2.0, 2.0)
                )
                
                # Generate
                self._omap_generator.generate3d()
                
                # Get Points (These are in World Frame)
                raw_points = self._omap_generator.get_occupied_positions()
                
                if len(raw_points) > 0:
                    points_np = np.array(raw_points)
                    
                    # FILTER: Disregard points over the robot
                    dist_to_robot = np.linalg.norm(points_np - current_pos_np, axis=1)
                    mask = dist_to_robot > 0.6
                    filtered_points = points_np[mask]

                    # Draw
                    if len(filtered_points) > 0:
                        points_list = [tuple(p) for p in filtered_points]
                        colors = [(1, 0, 0, 1)] * len(points_list) # Red
                        sizes = [10.0] * len(points_list)
                        
                        self._debug_draw.draw_points(points_list, colors, sizes)
                
                # Update the last position tracker
                self._last_omap_pos = current_pos_np

    def _update_sensors(self, step):
        # IMU UPDATE (Fast - 200 Hz)
        if self._IMU is not None:
            self._IMU_reading = self._IMU.get_imu_data()
            if self._data_collection_mode:
                self._IMU.log_data(
                    timestamp=self._time,
                    accel=self._IMU_reading['linear_acceleration'],
                    gyro=self._IMU_reading['angular_velocity']
                )

        # CAMERA UPDATE (Slow - 20 Hz)
        if not hasattr(self, '_last_cam_time'):
            self._last_cam_time = 0.0
        
        if self._cam is not None:
            if (self._time - self._last_cam_time) >= (1.0 / self._cam.get_frequency()):
                self._cam.render(sim_time=self._time)
                self._last_cam_time = self._time
        
        if self._sonar is not None:
            self._sonar.make_sonar_data()
            
        if self._DVL is not None:
            new_dvl_reading = self._DVL.get_linear_vel_fd(step)
            if not np.any(np.isnan(new_dvl_reading)):
                 self._DVL_reading = new_dvl_reading
                 self._DVL.publish_ros2(self._time, self._DVL_reading)
                 if self._data_collection_mode:
                     self._DVL.log_data(self._time, self._DVL_reading)

        # BARO UPDATE (Fast - 200 Hz)
        if self._baro is not None:
            self._baro_reading = self._baro.get_pressure()
            self._baro.publish_ros2(self._time, self._baro_reading)
            if self._data_collection_mode:
                self._baro.log_data(self._time, self._baro_reading)

        # Ground Truth Logging
        if self._data_collection_mode and self._rob is not None and hasattr(self, '_gt_csv_writer') and self._gt_csv_writer:
             try:
                 wt = omni.usd.get_world_transform_matrix(self._rob)
                 t = wt.ExtractTranslation()
                 q = wt.ExtractRotationQuat()
                 row = [self._time, t[0], t[1], t[2], q.GetReal(), q.GetImaginary()[0], q.GetImaginary()[1], q.GetImaginary()[2]]
                 self._gt_csv_writer.writerow(row)
             except Exception as e:
                 pass 

    def _handle_manual_control(self, step: float):
        if self._ctrl_mode=="Manual control" or self._ctrl_mode=="ROS + Manual control":
            # Get Keyboard inputs
            kb_force = self._force_cmd._base_command
            kb_torque = self._torque_cmd._base_command
            
            # Get Joystick inputs
            joy_force = self._joy_force._base_command
            joy_torque = self._joy_torque._base_command

            # Combine them (Summing them allows using both simultaneously)
            total_force = kb_force + joy_force
            total_torque = kb_torque + joy_torque

            user_is_controlling = np.linalg.norm(total_force) > 0.001 or np.linalg.norm(total_torque) > 0.001

            if self._ctrl_mode=="ROS + Manual control" and self._ros2_control_receiver is not None and not user_is_controlling:
                    self._ros2_control_receiver.update_control(step)

            if user_is_controlling:
                force_cmd = Gf.Vec3f(*total_force)
                torque_cmd = Gf.Vec3f(*total_torque)
                self._rob_forceAPI.CreateForceAttr().Set(force_cmd)
                self._rob_forceAPI.CreateTorqueAttr().Set(torque_cmd)

                msg = Wrench()
                msg.force.x = force_cmd[0]
                msg.force.y = force_cmd[1]
                msg.force.z = force_cmd[2]
                msg.torque.x = torque_cmd[0]
                msg.torque.y = torque_cmd[1]
                msg.torque.z = torque_cmd[2]
                self._rob_cmd_pub.publish(msg)
            else:
                self._rob_forceAPI.CreateForceAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                self._rob_forceAPI.CreateTorqueAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
                if self._ctrl_mode == "ROS + Manual control" and self._ros2_control_receiver is not None:
                     self._ros2_control_receiver.update_control(step)

    def _handle_waypoints_control(self, step: float):
        if self.waypoints_control_speed:
            SPEED = 1.0  # m/s
            ROT_SPEED = 1.0 # rad/s

            if len(self.waypoints) > 0:
                target_data = self.waypoints[0]
                target_pos = Gf.Vec3d(target_data[0], target_data[1], target_data[2])
                # Gf.Quatd expects (w, x, y, z)
                target_rot = Gf.Quatd(target_data[6], target_data[3], target_data[4], target_data[5])

                current_pos_attr = self._rob.GetAttribute('xformOp:translate')
                current_rot_attr = self._rob.GetAttribute('xformOp:orient')
                
                current_pos = current_pos_attr.Get()
                current_rot = current_rot_attr.Get()
                
                distance_vector = target_pos - current_pos
                distance = distance_vector.GetLength()

                max_move_this_frame = SPEED * step

                if distance <= max_move_this_frame:
                    new_pos = target_pos
                    position_reached = True
                else:
                    # Move exactly max_move_this_frame meters toward the target
                    direction = distance_vector / distance # Normalize vector
                    new_pos = current_pos + (direction * max_move_this_frame)
                    position_reached = False

                # Calculate dot product to find the angle between current and target quaternions
                dot = (current_rot.GetReal() * target_rot.GetReal() +
                        current_rot.GetImaginary()[0] * target_rot.GetImaginary()[0] +
                        current_rot.GetImaginary()[1] * target_rot.GetImaginary()[1] +
                        current_rot.GetImaginary()[2] * target_rot.GetImaginary()[2])
                
                # Keep dot product safely in bounds [-1, 1] to avoid math errors
                dot = max(-1.0, min(1.0, dot))
                
                # Calculate actual angular distance in radians (using absolute dot for shortest path)
                angle_diff = 2.0 * np.arccos(abs(dot))
                max_rot_this_frame = ROT_SPEED * step

                if angle_diff <= max_rot_this_frame:
                    # We will reach or pass the target rotation this frame. Snap perfectly to it.
                    new_rot = target_rot
                    rotation_reached = True
                else:
                    # Calculate exactly what percentage of the remaining angle we can cover this frame
                    slerp_amount = max_rot_this_frame / angle_diff
                    new_rot = Gf.Slerp(slerp_amount, current_rot, target_rot)
                    rotation_reached = False

                current_pos_attr.Set(new_pos)
                current_rot_attr.Set(new_rot)
                
                # Only move to the next waypoint if we have actually arrived
                if position_reached and rotation_reached:
                    self.waypoints.pop(0)
            else:
                print('Waypoints finished')
        else:
            if len(self.waypoints) > 0:
                waypoints = self.waypoints[0]
                self._rob.GetAttribute('xformOp:translate').Set(Gf.Vec3f(waypoints[0], waypoints[1], waypoints[2]))
                self._rob.GetAttribute('xformOp:orient').Set(Gf.Quatd(waypoints[3], waypoints[4], waypoints[5], waypoints[6]))
                self.waypoints.pop(0)
            else:
                print('Waypoints finished')
                self.generate_random_waypoints()
        
    def update_scenario(self, step: float):
        if not self._running_scenario:
            return
        
        self._time += step

        if hasattr(self, '_clock_pub') and self._clock_pub is not None:
            sim_time = self._timeline.get_current_time()
            clock_msg = Clock()
            clock_msg.clock.sec = int(sim_time)
            clock_msg.clock.nanosec = int((sim_time - int(sim_time)) * 1e9)
            self._clock_pub.publish(clock_msg)

        # Debug: Check actual update rate
        if not hasattr(self, '_last_update_time'):
            self._last_update_time = time.time()
            self._update_count = 0
        
        self._update_count += 1
        if self._update_count % 100 == 0:  # Print every 100 updates
            current_time = time.time()
            actual_hz = 100 / (current_time - self._last_update_time)
            print(f"Physics callback rate: {actual_hz:.1f} Hz")
            self._last_update_time = current_time

        # Update Robot Pose/Path and Map
        if self._publish_pose:
            current_pos_np, trans = self._publish_robot_pose_and_path()
            
            if self._publish_map:
                self._update_occupancy_map(current_pos_np, trans)

        # Update Sensors
        self._update_sensors(step)

        # Control Logic
        if self._ctrl_mode=="Manual control" or self._ctrl_mode=="ROS + Manual control":
            self._handle_manual_control(step)

        elif self._ctrl_mode=="Waypoints":
            self._handle_waypoints_control(step)
              
        elif self._ctrl_mode=="Straight line":
            SingleRigidPrim(prim_path=get_prim_path(self._rob)).set_linear_velocity(np.array([0.5,0,0])) 
        
        elif self._ctrl_mode=="ROS control":
            if self._ros2_control_receiver is not None:
                self._ros2_control_receiver.update_control(step)
            else:
                print("[Scenario] ROS2 Control receiver is not initialized, skipping update.")




        

        


