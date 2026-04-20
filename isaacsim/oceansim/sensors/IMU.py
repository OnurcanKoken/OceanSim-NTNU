from isaacsim.sensors.physics import IMUSensor
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3, PoseStamped
import numpy as np
import rclpy
import carb
import time
import yaml
import os
import omni
import omni.usd

class IMU(IMUSensor):
    def __init__(self,
    prim_path,
    name="imu",
    frequency=200, # or, dt=1./60
    translation=None, # or, position=np.array([0, 0, 0]),
    orientation=None,
    linear_acceleration_filter_size = 10,
    angular_velocity_filter_size = 10,
    orientation_filter_size = 10,
    config_path=None
    ):
        """Initialize an IMU sensor with configurable parameters.

        .. note::

            This class is inherited from ``IMUSensor``.

        Args:
            prim_path (str): prim path of the Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "imu".
            frequency (int, optional): update frequency of the IMU sensor in Hz. Defaults to 60.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            linear_acceleration_filter_size (int, optional): Size of the moving average filter for linear acceleration. Defaults to 10.
            angular_velocity_filter_size (int, optional): Size of the moving average filter for angular velocity. Defaults to 10.
            orientation_filter_size (int, optional): Size of the moving average filter for orientation. Defaults to 10.
        """
        super().__init__(
            prim_path=prim_path,
            name=name,
            frequency=frequency,
            translation=translation,
            orientation=orientation,
            linear_acceleration_filter_size=linear_acceleration_filter_size,
            angular_velocity_filter_size=angular_velocity_filter_size,
            orientation_filter_size=orientation_filter_size
        )

        self._frequency = frequency # Store frequency for metadata saving

        # Default Parameters (will be overwritten if config_path is valid)
        self.gyro_noise_density = 0.00016
        self.accel_noise_density = 0.0028
        self.gyro_walk = 0.000022
        self.acc_walk = 0.00086
        
        # Initialize Bias accumulation vectors
        self.init_gyro_bias = np.zeros(3)
        self.init_accel_bias = np.zeros(3)
        self.current_gyro_bias = np.zeros(3)
        self.current_accel_bias = np.zeros(3)

        # Track last time bias was updated to prevent double-update in single step
        self._last_bias_update_time = -1.0

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        elif config_path:
             carb.log_warn(f"[{self.name}] Config file not found at {config_path}. Using Valid Defaults.")
             print(f"[{self.name}] Config file not found at {config_path}. Using Valid Defaults.")

        self._csv_file = None
        self._csv_writer = None

    def load_config(self, path):
        try:
            with open(path, 'r') as file:
                config = yaml.safe_load(file)
                
            if 'accelerometer' in config:
                acc = config['accelerometer']
                self.accel_noise_density = acc.get('noise_density', self.accel_noise_density)
                self.acc_walk = acc.get('random_walk', self.acc_walk)
                init_bias = acc.get('bias_init', [0.0, 0.0, 0.0])
                self.init_accel_bias = np.array(init_bias, dtype=np.float64)
                self.current_accel_bias = np.copy(self.init_accel_bias)

            if 'gyroscope' in config:
                gyro = config['gyroscope']
                self.gyro_noise_density = gyro.get('noise_density', self.gyro_noise_density)
                self.gyro_walk = gyro.get('random_walk', self.gyro_walk)
                init_bias = gyro.get('bias_init', [0.0, 0.0, 0.0])
                self.init_gyro_bias = np.array(init_bias, dtype=np.float64)
                self.current_gyro_bias = np.copy(self.init_gyro_bias)

            print(f"[{self.name}] Loaded configuration from {path}")
            print(f"   Accel Noise: {self.accel_noise_density}, Walk: {self.acc_walk}, Bias: {self.current_accel_bias}")
            print(f"   Gyro Noise: {self.gyro_noise_density}, Walk: {self.gyro_walk}, Bias: {self.current_gyro_bias}")

        except Exception as e:
            carb.log_error(f"[{self.name}] Failed to load config: {e}")
            print(f"[{self.name}] Failed to load config: {e}")
        
        # Helper to track time for integration
        # self.last_step_time = None

    def _add_noise(self, true_ang_vel, true_lin_acc, dt, current_time):
        """
        Adds Gaussian white noise and Random Walk Bias to the raw sensor data.
        Guards against multiple updates in the same simulation step.
        """
        # Update Random Walk Bias (Wiener process)
        # Verify we haven't already updated for this timestep
        if current_time > self._last_bias_update_time:
            # Bias changes by: RandomStep * sqrt(dt)
            self.current_gyro_bias += np.random.normal(0, self.gyro_walk, 3)
            self.current_accel_bias += np.random.normal(0, self.acc_walk, 3)
            
            self._last_bias_update_time = current_time

        # Apply: Measurement = True + Bias + WhiteNoise
        noisy_ang_vel = true_ang_vel + self.current_gyro_bias + np.random.normal(0, self.gyro_noise_density, 3)
        noisy_lin_acc = true_lin_acc + self.current_accel_bias + np.random.normal(0, self.accel_noise_density, 3)

        return noisy_ang_vel, noisy_lin_acc

    def get_imu_data(self):
        """
        Retrieve the current IMU data including linear acceleration, angular velocity, and orientation.
        Returns:
            dict: A dictionary containing 'linear_acceleration', 'angular_velocity', and 'orientation'"""
        raw = self.get_current_frame(read_gravity=True)

        if self._frequency > 0:
            dt = 1.0 / self._frequency
        else:
            dt = 0.005 # Fallback default 200Hz

        current_time = raw.get('time')

        clean_ang_vel = np.array(raw['ang_vel'])
        clean_lin_acc = np.array(raw['lin_acc'])

        # Apply noise function
        noisy_ang_vel, noisy_lin_acc = self._add_noise(clean_ang_vel, clean_lin_acc, dt, current_time)

        imu_data = {
            'linear_acceleration': noisy_lin_acc,
            'linear_acceleration_covariance': self._linear_acceleration_covariance,
            'angular_velocity':    noisy_ang_vel,
            'angular_velocity_covariance': self._angular_velocity_covariance,
            'orientation':         np.array(raw['orientation']),
            'orientation_covariance': self._orientation_covariance,
            'time':                raw['time'],
            'physics_step':        raw['physics_step'],
        }

        if self._enable_ros2_pub:
            self._ros2_publish_imu(imu_data)
        else:
            carb.log_error('Not publishing IMU data')

        return imu_data

    def initialize(self,
                    enable_ros2_pub=True,
                    imu_topic="/oceansim/robot/imu",
                    pose_topic="/oceansim/robot/imu_pose",
                    pose_frame_id="mimosa_map",
                    ros2_pub_frequency=200,
                    orientation_covariance=np.zeros((3,3)).reshape(-1) * 1e-6,
                    angular_velocity_covariance=np.zeros((3,3)).reshape(-1) * 1e-4,
                    linear_acceleration_covariance=np.zeros((3,3)).reshape(-1) * 1e-3,
                ):
        
        carb.log_info('Initializing IMU...')

        # Reset Bias to Initial Configured Values
        self.current_gyro_bias = np.copy(self.init_gyro_bias)
        self.current_accel_bias = np.copy(self.init_accel_bias)
        self._last_bias_update_time = -1.0

        self._orientation_covariance = orientation_covariance
        self._angular_velocity_covariance = angular_velocity_covariance
        self._linear_acceleration_covariance = linear_acceleration_covariance

        # ROS2 configuration
        self._enable_ros2_pub = enable_ros2_pub
        self._imu_topic = imu_topic
        self._pose_topic = pose_topic
        self._pose_frame_id = pose_frame_id
        self._last_publish_time = 0.0
        self._ros2_pub_frequency = ros2_pub_frequency     # publish frequency, hz
        self._pose_pub = None
        self._setup_ros2_publisher()
        
        carb.log_info(f'[{self.name}] Initialized successfully.')

    def _setup_ros2_publisher(self):
        '''
        Setup the publisher for IMU
        '''
        try:
            if not self._enable_ros2_pub:
                return
            
            # Initialize ROS2 context if not already done
            if not rclpy.ok():
                rclpy.init()
                print(f'[{self.name}] ROS2 context initialized')

            # Create IMU publisher node
            node_name = f'oceansim_rob_imu_pub_{self.name.lower()}'.replace(' ', '_')
            self._ros2_imu_node = rclpy.create_node(node_name)
            self._imu_pub = self._ros2_imu_node.create_publisher(
                Imu,
                self._imu_topic,
                10
            )
            self._pose_pub = self._ros2_imu_node.create_publisher(
                PoseStamped,
                self._pose_topic,
                10
            )

        except Exception as e:
            print(f'[{self.name}] ROS2 IMU publisher setup failed: {e}')

    def _ros2_publish_imu(self, imu_data):
        """
        publish the IMU data
        """
        try:
            if self._imu_pub is None:
                return

            # fps control
            # current_time = time.time()
            # if current_time - self._last_publish_time < (1.0 / self._frequency):
            #     return

            # Create a ROS2 Imu message
            msg = Imu()
            # msg.header.stamp = self._ros2_imu_node.get_clock().now().to_msg()
            sim_time_sec = imu_data['time']  # This comes from IsaacSim
            msg.header.stamp.sec = int(sim_time_sec)
            msg.header.stamp.nanosec = int((sim_time_sec - int(sim_time_sec)) * 1e9)
            msg.header.frame_id = 'imu'
            orientation = imu_data['orientation']
            msg.orientation = Quaternion(
                x=float(orientation[0]),
                y=float(orientation[1]),
                z=float(orientation[2]),
                w=float(orientation[3])
            )
            msg.orientation_covariance = imu_data['orientation_covariance'].tolist()
            ang_vel = imu_data['angular_velocity']
            msg.angular_velocity = Vector3(
                x=float(ang_vel[0]),
                y=float(ang_vel[1]),
                z=float(ang_vel[2])
            )
            msg.angular_velocity_covariance = imu_data['angular_velocity_covariance'].tolist()
            lin_acc = imu_data['linear_acceleration']
            msg.linear_acceleration = Vector3(
                x=float(lin_acc[0]),
                y=float(lin_acc[1]),
                z=float(lin_acc[2])
            )
            msg.linear_acceleration_covariance = imu_data['linear_acceleration_covariance'].tolist()
            
            # Publish the message
            self._imu_pub.publish(msg)

            # Publish world pose for RViz visualization. get_world_pose() returns
            # (position_xyz, orientation_wxyz) from the inherited XformPrim.
            if self._pose_pub is not None:
                pos, quat_wxyz = self.get_world_pose()
                pose_msg = PoseStamped()
                pose_msg.header.stamp = msg.header.stamp
                pose_msg.header.frame_id = self._pose_frame_id
                pose_msg.pose.position.x = float(pos[0])
                pose_msg.pose.position.y = float(pos[1])
                pose_msg.pose.position.z = float(pos[2])
                pose_msg.pose.orientation.w = float(quat_wxyz[0])
                pose_msg.pose.orientation.x = float(quat_wxyz[1])
                pose_msg.pose.orientation.y = float(quat_wxyz[2])
                pose_msg.pose.orientation.z = float(quat_wxyz[3])
                self._pose_pub.publish(pose_msg)

            rclpy.spin_once(self._ros2_imu_node, timeout_sec=0.0)

            self._last_publish_time = time.time()

            # debug
            # self._ros2_uw_imu_node.get_logger().info(
            #     f'Published image: encoding={msg.encoding}, '
            #     f'width={msg.width}, height={msg.height}, step={msg.step}, '
            #     f'data_size={len(msg.data)}'
            # )

        except Exception as e:
            carb.log_error(f'[{self.name}] ROS2 IMU publish failed: {e}')

    def save_metadata(self, save_path):
        """
        Save IMU metadata (parameters and transform) to a YAML file.
        """
        print(f"[{self.name}] Attempting to save metadata to {save_path}...")
        try:
            # 1. Get Transform (Relative to Parent/Robot)
            # We assume the IMU prim is a child of the robot, so local transform = relative transform
            from pxr import UsdGeom, Gf
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.prim_path)
            
            if not prim.IsValid():
                print(f"[{self.name}] Metadata Error: Prim at {self.prim_path} is invalid!")
                # Fallback to identity if prim is missing (shouldn't happen during run)
                transform_list = [[1.0 if i==j else 0.0 for j in range(4)] for i in range(4)]
            else:
                xform = UsdGeom.Xformable(prim)
                local_transform = xform.GetLocalTransformation() # Gf.Matrix4d
                
                # Convert Matrix4d to list of lists - using GetRow for safety w.r.t python bindings
                transform_list = []
                for i in range(4):
                    row = local_transform.GetRow(i)
                    transform_list.append([float(row[0]), float(row[1]), float(row[2]), float(row[3])])
            
            # 2. Collect Parameters

            # 2. Collect Parameters
            metadata = {
                "sensor_type": "IMU",
                "name": self.name,
                "update_rate": self._frequency,
                "ros_topic": self._imu_topic if self._enable_ros2_pub else None,
                "accelerometer": {
                    "noise_density": float(self.accel_noise_density),
                    "random_walk": float(self.acc_walk),
                    "bias_init": self.init_accel_bias.tolist()
                },
                "gyroscope": {
                    "noise_density": float(self.gyro_noise_density),
                    "random_walk": float(self.gyro_walk),
                    "bias_init": self.init_gyro_bias.tolist()
                },
                "transform_to_body": transform_list
            }

            # 3. Write to File
            file_path = os.path.join(save_path, "imu_metadata.yaml")
            with open(file_path, 'w') as f:
                yaml.dump(metadata, f, default_flow_style=None)
            
            print(f"[{self.name}] Saved metadata to {file_path}")
            
        except Exception as e:
            carb.log_error(f"[{self.name}] Failed to save metadata: {e}")
            print(f"[{self.name}] Failed to save metadata: {e}")

    def init_logging(self, save_path):
        """Initialize CSV logging"""
        try:
            import csv
            file_path = os.path.join(save_path, "imu_data.csv")
            self._csv_file = open(file_path, mode='w', newline='')
            self._csv_writer = csv.writer(self._csv_file)
            # Header: Time, Accel(x,y,z), Gyro(x,y,z)
            self._csv_writer.writerow(['timestamp', 'a_x', 'a_y', 'a_z', 'g_x', 'g_y', 'g_z'])
            print(f"[{self.name}] Initialized data logging to {file_path}")
        except Exception as e:
            carb.log_error(f"[{self.name}] Failed to init logging: {e}")
            print(f"[{self.name}] Failed to init logging: {e}")

    def log_data(self, timestamp, accel, gyro):
        """Write a single frame of data to CSV"""
        if self._csv_writer:
            try:
                # Expects numpy arrays for accel and gyro
                row = [timestamp, accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]]
                self._csv_writer.writerow(row)
            except Exception as e:
                print(f"[{self.name}] Error writing log: {e}")

    def close(self):
        """Cleanup ROS2 node and publisher, and close log file"""
        try:
            # Close CSV
            if self._csv_file:
                self._csv_file.close()
                self._csv_file = None
                print(f"[{self.name}] Closed log file.")

            if self._enable_ros2_pub:
                # Destroy publisher
                if self._imu_pub is not None:
                    self._ros2_imu_node.destroy_publisher(self._imu_pub)
                    self._imu_pub = None

                if self._pose_pub is not None:
                    self._ros2_imu_node.destroy_publisher(self._pose_pub)
                    self._pose_pub = None

                # Destroy node
                if self._ros2_imu_node is not None:
                    self._ros2_imu_node.destroy_node()
                    self._ros2_imu_node = None
                
            print(f'[{self.name}] ROS2 Node cleaned up.')
        except Exception as e:
            print(f'[{self.name}] Error cleaning up ROS2 node: {e}')
            
# usage example:
# imu_sensor = IMU(prim_path="/World/Robot/IMU", name="imu", frequency=100)
# imu_data = imu_sensor.get_imu_data()