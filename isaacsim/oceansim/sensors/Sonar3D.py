from isaacsim.sensors.rtx import LidarRtx
import omni.replicator.core as rep
import rclpy
from sensor_msgs.msg import PointCloud2

class Sonar3D(LidarRtx):
    def __init__(self, 
                 prim_path, 
                 name="Sonar3D", 
                 translation=None, 
                 orientation=None, 
                 config_file_name=None,
                 **sensor_atributes):
        
        self._prim_path = prim_path
        self._name = name

        super().__init__(prim_path=prim_path,
                         name=name,
                         translation=translation,
                         orientation=orientation,
                         config_file_name=config_file_name,
                         **sensor_atributes)
    
    def custom_init(self, enable_ros2_pub=True):

        self._enable_ros2_pub = enable_ros2_pub
        
        # See pointcloud in simulator
        self.enable_visualization()

        # Attach annotator to collect Lidar data with timestamp
        self.attach_annotator("IsaacCreateRTXLidarScanBuffer", 
                              outputTimestamp=True,
                              outputEmitterId=True,
                              outputAzimuth=True)
        
        # Create custom ros2 node for lidar data with timestamp
        self._setup_ros2_publisher()


        # FOR STANDARD ROS2 BRIDGE OF LIDAR DATA
        # # The class automatically makes a render product 
        # self._render_product_path = self.get_render_product_path()
        # # This writer handles publishing the data to ROS2 topic
        # self.writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        # # Define the topic name that you see in the ros2 topics list and set frame ID to "map" so you can visualize it in Rviz2
        # self.writer.initialize(topicName="sonar3D_point_cloud", frameId="map")
        # # Attach the render product of the lidar to this writer
        # self.writer.attach([self._render_product_path])

    def _setup_ros2_publisher(self):
        try:
            if not self._enable_ros2_pub:
                return
            
            # Initialize ROS2 context if not already done
            if not rclpy.ok():
                rclpy.init()
                print(f'[{self._name}] ROS2 context initialized')

            # Create ROS2 Lidar publisher node
            node_name = f'oceansim_3D_sonar_pub_timestamp'
            self._ros2_sonar3D_node = rclpy.create_node(node_name)
            self._sonar3D_pub = self._ros2_sonar3D_node.create_publisher(
                PointCloud2,
                "/oceansim/sonar3d",
                10
            )

        except Exception as e:
            print(f'[{self._name}] ROS2 uw image publisher setup failed: {e}')

    def _ros2_publish_sonar3D(self, pointcloud, timestamp):
        """
        Publish rtx lidar data to ros2 topic

        pointcloud with timestamp come from annotator
        """

        try:
            if self._sonar3D_pub is None:
                return
            
            print("in publisher")

            msg = PointCloud2
            
            rclpy.spin_once(self._ros2_sonar3D_node, timeout_sec=0.0)

        except Exception as e:
            print(f'[{self._name}] ROS2 uw image publish failed: {e}')    


    def close(self):
        # These writers and annotators used memory and you need to release them
        self.detach_all_annotators() 
        self.detach_all_writers()

