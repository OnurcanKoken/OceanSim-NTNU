import time
import os
import numpy as np
from enum import Enum

from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_path
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist, Wrench, PoseStamped
from std_msgs.msg import Header

'''
Attention:

Before OceanSim extension being activated, the extension isaacsim.ros2.bridge should be activated, otherwise rclpy will
fail to be loaded.

so, we suggest that make sure the extension isaacsim.ros2.bridge is being setup to "AUTOLOADED" in Window->Extension.
'''
import rclpy

try:
    from pxr import Gf, PhysxSchema
    PXR_AVAILABLE = True
except ImportError:
    PXR_AVAILABLE = False
    Gf = None 
    PhysxSchema = None

ROS2_AVAILABLE = False

class ROS2_CONTROL_MODE(Enum):
    VEL = 1     # velocity control mode
    FORCE = 2   # force control mode
    WAYPOINT = 3 # waypoint control mode

class ROS2ControlReceiver:
    """
    ROS2 Control Receiver
    
    for recieving velocity and force command
    """
    
    def __init__(self, robot_prim, name="ROS2ControlReceiver"):
        """
        initialize ROS2 Control Receiver
        
        Args:
            robot_prim: robot prim path
            name (str): receiver name
        """
        self._name = name
        self._robot_prim = robot_prim
        
        # configuration
        self._enable_ros2 = False
        
        self._ros2_control_mode = ROS2_CONTROL_MODE.VEL  # control mode
        self._ros2_vel_node = None
        self._ros2_force_node = None
        self._ros2_waypoint_node = None
        
        # command cache
        self.force_cmd = [0.0, 0.0, 0.0]
        self.torque_cmd = [0.0, 0.0, 0.0]
        self.linear_vel = [0.0, 0.0, 0.0]
        self.angular_vel = [0.0, 0.0, 0.0]
        self.last_command_time = time.time()
        self.command_timeout = 2.0
        self._update_count = 0
        self.waypoint = None
        
        # Physics API - using scenario.py created instance
        self._force_api = None
        self._scenario_force_api = None 
        
        print(f"[{self._name}] Initialized for robot prim")
        
    def initialize(self, enable_ros2=True, vel_topic="/oceansim/robot/vel_cmd", force_topic="/oceansim/robot/force_cmd", waypoint_topic="/aeplanner/nbv_setpoint"):
        """
        initialize reciever function
        
        Args:
            enable_ros2 (bool): whether using ros2
            vel_topic (str): topic name of vel
            force_topic (str): topic name of force(include torque)
        """
        self._enable_ros2 = enable_ros2
        self._vel_topic = vel_topic
        self._force_topic = force_topic
        self._waypoint_topic = waypoint_topic
        
        if not self._enable_ros2:
            print(f'[{self._name}] ROS2 disabled by configuration')
            return
        
        self._setup_subscriber()
        self._setup_physics()
        
        print(f'[{self._name}] Control Receiver Initialized:')
        print(f'[{self._name}] ROS2 Bridge: {self._enable_ros2}')
        if PXR_AVAILABLE and self._robot_prim:
            print(f'[{self._name}] Robot Prim: {self._robot_prim.GetPath()}')
        else:
            print(f'[{self._name}] Robot Prim: {self._robot_prim}')

    def set_scenario_force_api(self, scenario_force_api):
        """
        setting the force api
        """
        self._scenario_force_api = scenario_force_api
        
    def _setup_physics(self):
        """
        setting the physics control API(PXR)
        """
        if not PXR_AVAILABLE:
            print(f'[{self._name}] PXR not available, physics API disabled')
            return
            
        try:
            if self._scenario_force_api is not None:
                self._force_api = self._scenario_force_api
            else:
                if self._robot_prim.HasAPI(PhysxSchema.PhysxForceAPI):
                    self._force_api = PhysxSchema.PhysxForceAPI(self._robot_prim)
                else:
                    self._force_api = PhysxSchema.PhysxForceAPI.Apply(self._robot_prim)
                
        except Exception as e:
            print(f'[{self._name}] Physics API set failed: {e}')
    
    def _setup_subscriber(self):
        """
        setting the ROS2 subscriber
        """
        try:
            # import ROS2 module
            from sensor_msgs.msg import Image
            from geometry_msgs.msg import Twist, Wrench
            from std_msgs.msg import Header
            
            # Initialize ROS2 context if not already done
            if not rclpy.ok():
                rclpy.init()
                print(f'[{self._name}] ROS2 context initialized')
            
            # Create velocity subscriber node
            node_name = f'oceansim_rob_velocity_control_{self._name.lower()}'.replace(' ', '_')
            self._ros2_vel_node = rclpy.create_node(node_name)
            self._ros2_vel_subscriber = self._ros2_vel_node.create_subscription(
                Twist,
                self._vel_topic,
                self._vel_callback,
                10
            )

            # Create force subscriber node
            node_name = f'oceansim_rob_force_control_{self._name.lower()}'.replace(' ', '_')
            self._ros2_force_node = rclpy.create_node(node_name)
            self._force_subscriber = self._ros2_force_node.create_subscription(
                Wrench,
                self._force_topic,
                self._force_callback,
                10
            )

            # Create waypoint subscriber node
            node_name = f'oceansim_rob_waypoint_control_{self._name.lower()}'.replace(' ', '_')
            self._ros2_waypoint_node = rclpy.create_node(node_name)
            self._waypoint_subscriber = self._ros2_waypoint_node.create_subscription(
                PoseStamped,
                self._waypoint_topic,
                self._waypoint_callback,
                10
            )
            
        except Exception as e:
            self._enable_ros2 = False

    def _setup_ros2_control_mode(self, ctrl_mode):
        if ctrl_mode == "velocity control":
            self._ros2_control_mode = ROS2_CONTROL_MODE.VEL
        elif ctrl_mode == "force control":
            self._ros2_control_mode = ROS2_CONTROL_MODE.FORCE
        elif ctrl_mode == "waypoint control":
            self._ros2_control_mode = ROS2_CONTROL_MODE.WAYPOINT
    
    def _vel_callback(self, msg):
        """
        msg type: geometry_msgs/Twist
        
        include linear and angular velocity
        """
        print(f'[{self._name}] recieve ROS2 msg, type: {type(msg).__name__}, linear: {msg.linear}, angular: {msg.angular}')
        
        if not self._enable_ros2:
            print(f'[{self._name}] ROS2 is not enabled, ignore msg')
            return
        
        try:
            current_time = time.time()
            
            self.linear_vel = [msg.linear.x, msg.linear.y, msg.linear.z]
            self.angular_vel = [msg.angular.x, msg.angular.y, msg.angular.z]
            self.last_command_time = current_time
            
            print(f'Received velocity - Linear: {self.linear_vel}, Angular: {self.angular_vel}')
            # self._update_receive_stats(current_time)
            
        except Exception as e:
            print(f'[{self._name}] Vel Receive Failed: {e}')
        
    def _force_callback(self, msg):
        """
        msg type: geometry_msgs/Wrench
        
        include force and torque
        """
        print(f'[{self._name}] recieve ROS2 msg, type: {type(msg).__name__}, force: {msg.force}, torque: {msg.torque}')

        if not self._enable_ros2:
            print(f'[{self._name}] ROS2 is not enabled, ignore msg')
            return

        try:
            current_time = time.time()

            self.force_cmd = [msg.force.x, msg.force.y, msg.force.z]  # force
            self.torque_cmd = [msg.torque.x, msg.torque.y, msg.torque.z]  # torque
            self.last_command_time = current_time

            print(f'Received force - Force: {self.force_cmd}, Torque: {self.torque_cmd}')

        except Exception as e:
            print(f'[{self._name}] force Receive Failed: {e}')

    def _waypoint_callback(self, msg):
        """
        msg type: geometry_msgs/PoseStamped
        """
        print(f'[{self._name}] received ROS 2 msg, type: {type(msg).__name__}, position: {msg.pose.position}, orientation: {msg.pose.orientation}')

        if not self._enable_ros2:
            print(f'[{self._name}] ROS 2 is not enabled, ignore msg')
            return

        current_time = time.time()

        self.waypoint = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]

        self.last_command_time = current_time

    def update_control(self, step: float):
        """
        update control
        
        this function will be called in each simulation step. ( in scenario.update_scenario() )
        """
        if not self._enable_ros2 or not self._ros2_vel_node or not self._ros2_force_node:
            return
        
        try:
            if self._ros2_control_mode == ROS2_CONTROL_MODE.VEL: # velocity mode
                self._update_count += 1

                if self._update_count % 10 == 0: # need delay, otherwise the scene will be blocked
                    self._update_count = 0

                    rclpy.spin_once(self._ros2_vel_node, timeout_sec=0.0)
                    
                    rigid_prim = SingleRigidPrim(prim_path=get_prim_path(self._robot_prim))
                    rigid_prim.set_linear_velocity(np.array([0.0, 0.0, 0.0]))  # reset
                    rigid_prim.set_angular_velocity(np.array([0.0, 0.0, 0.0]))  # reset
                    rigid_prim.set_linear_velocity(np.array(self.linear_vel))
                    rigid_prim.set_angular_velocity(np.array(self.angular_vel))

            elif self._ros2_control_mode == ROS2_CONTROL_MODE.FORCE: # force mode
                # using PXR API to contorl
                if PXR_AVAILABLE:
                    
                    self._update_count += 1

                    if self._update_count % 10 == 0: # need delay, otherwise the scene will be blocked
                        self._update_count = 0

                        rclpy.spin_once(self._ros2_force_node, timeout_sec=0.0)

                        force_gf = Gf.Vec3f(float(self.force_cmd[0]), float(self.force_cmd[1]), float(self.force_cmd[2]))
                        torque_gf = Gf.Vec3f(float(self.torque_cmd[0]), float(self.torque_cmd[1]), float(self.torque_cmd[2]))
                    
                        if self._force_api:
                            try:
                                self._force_api.CreateForceAttr().Set(force_gf)
                                self._force_api.CreateTorqueAttr().Set(torque_gf)
                            except Exception as e:
                                print(f'[{self._name}] Force API Update Failed: {e}')

            elif self._ros2_control_mode == ROS2_CONTROL_MODE.WAYPOINT: # waypoint mode
                rclpy.spin_once(self._ros2_waypoint_node, timeout_sec=0.0)

                SPEED = 1.0  # m/s
                ROT_SPEED = 1.0 # rad/s

                if self.waypoint != None:
                    target_data = self.waypoint
                    target_pos = Gf.Vec3d(target_data[0], target_data[1], target_data[2])
                    # Gf.Quatd expects (w, x, y, z)
                    target_rot = Gf.Quatd(target_data[6], target_data[3], target_data[4], target_data[5])

                    current_pos_attr = self._robot_prim.GetAttribute('xformOp:translate')
                    current_rot_attr = self._robot_prim.GetAttribute('xformOp:orient')
                    
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
                        self.waypoint = None
                else:
                    pass
                    # print('Waypoints finished')

                
        except Exception as e:
            print(f'[{self._name}] Control Update Failed: {e}')
    
    def close(self):
        # Clean up ROS2 resources
        if self._enable_ros2:
            if self._ros2_vel_node:
                self._ros2_vel_node.destroy_node()
            if self._ros2_force_node:
                self._ros2_force_node.destroy_node()

        self._update_count = 0
        self.force_cmd = [0.0, 0.0, 0.0]
        self.torque_cmd = [0.0, 0.0, 0.0]
        self.linear_vel = [0.0, 0.0, 0.0]
        self.angular_vel = [0.0, 0.0, 0.0]

        # rclpy.shutdown()

        print(f'[{self._name}] ROS2_Control_receiver closed.') 

