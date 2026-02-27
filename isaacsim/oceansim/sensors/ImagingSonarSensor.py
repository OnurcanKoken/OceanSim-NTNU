from isaacsim.sensors.camera import Camera
import omni.replicator.core as rep
import omni.ui as ui
import numpy as np
from omni.replicator.core.scripts.functional import write_np
import omni.timeline as timeline
import warp as wp
from isaacsim.oceansim.utils.ImagingSonar_kernels import *
import rclpy
import rclpy.time
from sensor_msgs.msg import Image
from blueye_interfaces.msg import FloatStamped

from pxr import Gf
import omni.kit.commands
import omni.graph.core as og
import carb
from isaacsim.core.prims import SingleXFormPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat

# Future TODO
# In future release, wrap this class around RTX lidar

class ImagingSonarSensor(Camera):
    def __init__(self, 
                 prim_path, 
                 name = "ImagingSonar", 
                 frequency = None, 
                 dt = None, 
                 position = None, 
                 orientation = None, 
                 translation = None, 
                 render_product_path = None,
                 physics_sim_view = None,
                 min_range: float = 0.1, # m
                 max_range: float = 40.0, # m
                 range_res: float = 0.0025, # deg
                 hori_fov: float = 130.0, # deg
                 vert_fov: float = 20.0, # deg
                 angular_res: float = 0.6, # deg
                 hori_res: int = 4000 # isaac camera render product only accepts square pixel, 
                                      # for now vertical res is automatically set with ratio of hori_fov vs.vert_fov 
                 ):
        
    
        """Initialize an imaging sonar sensor with physical parameters.
    
        Args:
            prim_path (str): prim path of the Camera Prim to encapsulate or create.
            name (str, optional): shortname to be used as a key by Scene class.
                                    Note: needs to be unique if the object is added to the Scene.
                                    Defaults to "ImagingSonar".
            frequency (Optional[int], optional): Frequency of the sensor (i.e: how often is the data frame updated).
                                                Defaults to None.
            dt (Optional[str], optional): dt of the sensor (i.e: period at which a the data frame updated). Defaults to None.
            resolution (Optional[Tuple[int, int]], optional): resolution of the camera (width, height). Defaults to None.
            position (Optional[Sequence[float]], optional): position in the world frame of the prim. shape is (3, ).
                                                        Defaults to None, which means left unchanged.
            translation (Optional[Sequence[float]], optional): translation in the local frame of the prim
                                                            (with respect to its parent prim). shape is (3, ).
                                                            Defaults to None, which means left unchanged.
            orientation (Optional[Sequence[float]], optional): quaternion orientation in the world/ local frame of the prim
                                                            (depends if translation or position is specified).
                                                            quaternion is scalar-first (w, x, y, z). shape is (4, ).
                                                            Defaults to None, which means left unchanged.
            render_product_path (str): path to an existing render product, will be used instead of creating a new render product
                                    the resolution and camera attached to this render product will be set based on the input arguments.
                                    Note: Using same render product path on two Camera objects with different camera prims, resolutions is not supported
                                    Defaults to None

            physics_sim_view (_type_, optional): _description_. Defaults to None.            
            min_range (float, optional): Minimum detection range in meters. Defaults to 0.2.
            max_range (float, optional): Maximum detection range in meters. Defaults to 3.0.
            range_res (float, optional): Range resolution in meters. Defaults to 0.008.
            hori_fov (float, optional): Horizontal field of view in degrees. Defaults to 130.0.
            vert_fov (float, optional): Vertical field of view in degrees. Defaults to 20.0.
            angular_res (float, optional): Angular resolution in degrees. Defaults to 0.5.
            hori_res (int, optional): Horizontal pixel resolution. Defaults to 3000.
    
        Note:
            - Vertical resolution is automatically calculated to maintain aspect ratio
            - Uses Warp for GPU-accelerated sonar image generation
            - Creates polar coordinate meshgrid for sonar returns processing
        """


        self._name = name
        # Raw parameters from Oculus M370s\MT370s\MD370s
        self.max_range = max_range # m (max is 200 m in datasheet )
        self.min_range = min_range # m (min is 0.2 m in datasheet)
        self.range_res = range_res # m (datasheet is 0.008 m)
        self.hori_fov = hori_fov # degree (hori_fov is 130 degrees in datasheet)
        self.vert_fov = vert_fov # degree (vert_fov is 20 degrees in datasheet)
        self.angular_res = angular_res # degree (datasheet is 2 deg)
        self.hori_res= hori_res

        # self.beam_separation = 0.5 # degree (Not USED FOR NOW)!!
        # self.num_beams = 256 # (max number of beams) (NOT USED FOR NOW)!!
        # self.update_rate = 40 # Hz (max update rate) (NOT USED FOR NOW)!!


        # Generate sonar map's r and z meshgrid
        self.min_azi = np.deg2rad(90-self.hori_fov/2)
        r, azi = np.meshgrid(np.arange(self.min_range,self.max_range,self.range_res),
                                       np.arange(np.deg2rad(90-self.hori_fov/2), np.deg2rad(90+self.hori_fov/2), np.deg2rad(self.angular_res)),
                                       indexing='ij')
        self.r = wp.array(r, shape=r.shape, dtype=wp.float32)
        self.azi = wp.array(azi, shape=r.shape, dtype=wp.float32)

        # Load array that doesn't change shapes to cuda for reusage memory
        # Users can also automatically see if they have set a reasonable parameter 
        # for sonar map bin size\resolution once load the sensor
        self.bin_sum = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.bin_count = wp.zeros(shape=self.r.shape, dtype=wp.int32)
        self.binned_intensity = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.sonar_map = wp.zeros(shape=self.r.shape, dtype=wp.vec3)
        self.sonar_image = wp.zeros(shape=(self.r.shape[0], self.r.shape[1], 4), dtype=wp.uint8)
        self.gau_noise = wp.zeros(shape=self.r.shape, dtype=wp.float32)
        self.range_dependent_ray_noise = wp.zeros(shape=self.r.shape, dtype=wp.float32)

        self.AR = self.hori_fov / self.vert_fov
        self.vert_res = int(self.hori_res / self.AR)
        # By doing this, I am assuming the vertical beam separation
        # is the same as the beam horizontal separation. 
        # This is bacause replicator raytracing is specified as resolutions
        # while non-squre pixel is not supported in Isaac sim. See details below.
        
        super().__init__(prim_path=prim_path, 
                         name=name, 
                         frequency=frequency,
                         dt=dt, 
                         resolution=[self.hori_res, self.vert_res],
                         position=position, 
                         orientation=orientation, 
                         translation=translation, 
                         render_product_path=render_product_path)

        self.set_clipping_range(
            near_distance=self.min_range,
            far_distance=self.max_range
        )
        # This is a bug. Needs to call initialize() before changing aperture
        # https://forums.developer.nvidia.com/t/error-when-setting-a-cameras-vertical-horizontal-aperture/271314
        # This line initialize the camera
        self.initialize(physics_sim_view)

        # Assume the default focal length to compute the desired horizontal aperture
        # The reason why we are doing this is because Isaac sim will fix vertical aperture
        # given aspect ratio for mandating square pixles
        # https://forums.developer.nvidia.com/t/how-to-modify-the-cameras-field-of-view/278427/5
        self.focal_length = self.get_focal_length()
        horizontal_aper = 2 * self.focal_length * np.tan(np.deg2rad(self.hori_fov) / 2)
        self.set_horizontal_aperture(horizontal_aper)
        # Notice if you would like to observe sonar view from linked viewport.
        # Only horizontal fov is displayed correctly while the vertical fov is
        # followed by your viewport aspect ratio settings.
        

    # Initialize the sensor so that annotator is 
    # loaded on cuda and ready to acquire data
    # Data is generated per simulation tick

    # do_array_copy: If True, retrieve a copy of the data array. 
    # This is recommended for workflows using asynchronous
    # backends to manage the data lifetime. 
    # Can be set to False to gain performance if the data is 
    # expected to be used immediately within the writer. Defaults to True.

    def sonar_initialize(self,output_dir : str = None, viewport: bool = True, include_unlabelled = False, if_array_copy: bool = True, enable_ros2_pub: bool = True, sonar_topic: str = '/oceansim/robot/imaging_sonar'):
        """Initialize sonar data processing pipeline and annotators.
    
        Args:
            output_dir (str, optional): Directory to save sonar data. Defaults to None.
                                        If set to None, sonar will not write data.
            viewport (bool, optional): Enable viewport visualization. Defaults to True.
                                        Set to False for Sonar running without visualization.
            include_unlabelled (bool, optional): Include unlabelled objects to be scanned into sonar view. Defaults to False.
            if_array_copy (bool, optional): If True, retrieve a copy of the data array. 
                                            This is recommended for workflows using asynchronous backends to manage the data lifetime. 
                                            Can be set to False to gain performance if the data is expected to be used immediately within the writer. 
                                            Defaults to True.
                                            
        Note:
            - Attaches pointcloud, camera params, and semantic segmentation annotators
            - Sets up Warp arrays for sonar image processing
            - Can optionally write data to disk if output_dir specified
        """
        self.writing = False
        self._viewport = viewport
        self._device = str(wp.get_preferred_device())
        self.scan_data = {}
        self.id = 0

        self.pointcloud_annot = rep.AnnotatorRegistry.get_annotator(
            name="pointcloud",
            init_params={"includeUnlabelled": include_unlabelled},
            do_array_copy=if_array_copy,
            device=self._device
            )
        
        # CameraParams returns a numpy-compatible dict — fetching on CPU avoids
        # a round-trip through the GPU annotator cache for no benefit.
        self.cameraParams_annot = rep.AnnotatorRegistry.get_annotator(
            name="CameraParams",
            do_array_copy=if_array_copy,
            device="cpu"
            )
        
        # semantic_segmentation is only used for idToLabels (a Python dict).
        # Fetching on CPU avoids needlessly copying a tiny dict through GPU memory.
        self.semanticSeg_annot = rep.AnnotatorRegistry.get_annotator(
            name='semantic_segmentation',
            init_params={"colorize": False},
            do_array_copy=if_array_copy,
            device="cpu"
        )

        print(f'[{self._name}] Using {self._device}' )
        print(f'[{self._name}] Render query res: {self.hori_res} x {self.vert_res}. Binning res: {self.r.shape[0]} x {self.r.shape[1]}')

        self.pointcloud_annot.attach(self._render_product_path)
        self.cameraParams_annot.attach(self._render_product_path)
        self.semanticSeg_annot.attach(self._render_product_path)
        
        if output_dir is not None:
            self.writing = True
            self.backend = rep.BackendDispatch({"paths": {"out_dir": output_dir}})
        if self._viewport:
            self.make_sonar_viewport()
        
        print(f'[{self._name}] Initialized successfully. Data writing: {self.writing}')

        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()
        self.sonar_map.zero_()
        self.sonar_image.zero_()
        self.range_dependent_ray_noise.zero_()
        self.gau_noise.zero_()

        # Pre-allocate per-frame GPU buffers at maximum possible size to avoid
        # per-frame alloc/free cycles. Each call to make_sonar_data() previously
        # created wp.empty() arrays sized to num_points (which varies per frame).
        # The async free of the previous frame's buffers raced with the next frame's
        # memcpy_d2h (from numpy()), corrupting the CUDA context (error 700).
        # By pre-allocating at max size and slicing, we eliminate all mid-run allocs.
        self._max_points = self.hori_res * self.vert_res
        self._intensity_buf = wp.zeros(shape=(self._max_points,), dtype=wp.float32)
        self._pcl_local_buf = wp.zeros(shape=(self._max_points,), dtype=wp.vec3)
        self._pcl_spher_buf = wp.zeros(shape=(self._max_points,), dtype=wp.vec3)
        print(f'[{self._name}] Pre-allocated per-frame GPU buffers for max {self._max_points} points.')

        # Pre-allocate maximum buffers used in normalizing step to avoid per-frame
        # wp.zeros() GPU allocations inside the hot path.
        self._maximum_all = wp.zeros(shape=(1,), dtype=wp.float32)
        self._maximum_range = wp.zeros(shape=(self.r.shape[0],), dtype=wp.float32)

        # Cache the last-seen idToLabels and its corresponding GPU array.
        # Labels rarely change at runtime; recomputing and re-uploading every frame
        # is pure waste. We only rebuild when the label dict actually changes.
        self._cached_idToLabels = None
        self._cached_indexToRefl = None

        # ROS2 configuration
        self._enable_ros2_pub = enable_ros2_pub
        self._sonar_topic = sonar_topic
        self._setup_ros2_publisher()


    def _setup_ros2_publisher(self):
        '''
        setup the publisher for the sonar data
        '''
        try:
            if not self._enable_ros2_pub:
                return

            # Initialize ROS2 context if not already done
            if not rclpy.ok():
                rclpy.init()
                print(f'[{self._name}] ROS2 context initialized')

            # Create sonar data publisher node
            import time
            unique_id = int(time.time()) % 1000
            node_name = f'oceansim_rob_sonar_pub_{self._name.lower()}_{unique_id}'.replace(' ', '_')
            self._ros2_sonar_node = rclpy.create_node(node_name)
            self._sonar_pub = self._ros2_sonar_node.create_publisher(
                Image, 
                self._sonar_topic, 
                10
            )

            self._pitch_pub = self._ros2_sonar_node.create_publisher(
                FloatStamped,
                "/oceansim/robot/sonar_pitch", 
                10
            )
        
        except Exception as e:
            print(f'[{self._name}] ROS2 sonar data publisher setup failed: {e}')

    def publish_pitch(self):
        """Calculates and publishes the pitch of the sonar relative to its parent (/World/rob)."""
        if not hasattr(self, '_pitch_pub') or self._pitch_pub is None:
            return

        try:
            # Get the local transform matrix relative to the parent prim
            local_transform = self.get_local_pose() # Returns (translation, orientation)
            
            # Extract the orientation (quaternion)
            # local_transform[1] is the quaternion in (w, x, y, z) format
            quat = Gf.Quatd(float(local_transform[1][0]), 
                            float(local_transform[1][1]), 
                            float(local_transform[1][2]), 
                            float(local_transform[1][3]))
            
            # Convert Quaternion to Rotation Matrix and Decompose to Euler angles
            rotation = Gf.Rotation(quat)
            euler_angles = rotation.Decompose(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0), Gf.Vec3d(0, 0, 1))
            
            # Create and populate the ROS2 message
            msg = FloatStamped()
            
            sim_time = timeline.get_timeline_interface().get_current_time()
            msg.header.stamp.sec = int(sim_time)
            msg.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1e9)
            msg.header.frame_id = "sonar_link"            
            msg.data = float(euler_angles[1])
            
            self._pitch_pub.publish(msg)
            
            # Process discovery/event queue
            if rclpy.ok():
                rclpy.spin_once(self._ros2_sonar_node, timeout_sec=0)

        except Exception as e:
            print(f"[{self._name}] Local pitch publication error: {e}")

    def scan(self):

        """Capture a single sonar scan frame and store the raw data.
    
        Returns:
            bool: True if scan was successful (valid data received), False otherwise
    
        Note:
            - Stores pointcloud, normals, semantics, and camera transform in scan_data dict
            - First few frames may be empty due to CUDA initialization
            - Automatically skips frames with no detected objects
        """
        # # Due to the time to load annotator to cuda, the first few simulation tick gives no annotation in memory.
        # # This would also reult error when no mesh within the sonar fov
        # # NOTE: Isaac Sim annotator output has squeezed the first dimention after 5.0 update: (1,N,3) -> (N,3)   
        # if len(self.semanticSeg_annot.get_data()['info']['idToLabels']) !=0:
        #     self.scan_data['pcl'] = self.pointcloud_annot.get_data(device=self._device)['data']  # shape :(N,3) <class 'warp.types.array'>
        #     self.scan_data['normals'] = self.pointcloud_annot.get_data(device=self._device)['info']['pointNormals'] # shape :(N,4) <class 'warp.types.array'>
        #     self.scan_data['semantics'] = self.pointcloud_annot.get_data(device=self._device)['info']['pointSemantic'] # shape: (N) <class 'warp.types.array'>
        #     self.scan_data['viewTransform'] = self.cameraParams_annot.get_data()['cameraViewTransform'].reshape(4,4).T # 4 by 4 np.ndarray extrinsic matrix
        #     self.scan_data['idToLabels'] = self.semanticSeg_annot.get_data()['info']['idToLabels'] # dict 
        #     return True
        # else:
        #     return False
        try:
            # 1. Check if the pipeline is actually ready to give us data.
            # We access the semantic annotator first. If the pipeline is cold, 
            # this (or the pointcloud access) will throw the KeyError.
            seg_data = self.semanticSeg_annot.get_data()
            
            # If we got here, the annotator dictionary exists, but we should check if data is populated
            if not seg_data or 'info' not in seg_data:
                return False

            # 2. Proceed with your existing logic
            # NOTE: Isaac Sim annotator output has squeezed the first dimension after 5.0 update
            if len(seg_data['info']['idToLabels']) != 0:
                # Wrap the pointcloud access as well, just to be safe
                pcl_data = self.pointcloud_annot.get_data(device=self._device)
                
                self.scan_data['pcl'] = pcl_data['data']
                self.scan_data['normals'] = pcl_data['info']['pointNormals']
                self.scan_data['semantics'] = pcl_data['info']['pointSemantic']
                
                self.scan_data['viewTransform'] = self.cameraParams_annot.get_data()['cameraViewTransform'].reshape(4,4).T
                self.scan_data['idToLabels'] = seg_data['info']['idToLabels']
                return True
            else:
                return False

        except KeyError as e:
            # This catches the specific '/Render/PostProcess/SDGPipeline/...' error
            # It implies the Render Product hasn't completed a full cycle yet.
            # We silently return False so the simulation can keep ticking until it's ready.
            return False
        except Exception as e:
            # Catch other unforeseen errors to keep the sim alive
            print(f"[{self._name}] Scan failed with unexpected error: {e}")
            return False

    def _ros2_publish_sonar_image(self, sonar_data, frame_id="sonar_link"):
        '''
        Publish the sonar data as a ROS2 Image message
        '''
        try:
            if not self._enable_ros2_pub:
                return

            # numpy() returns a C-contiguous CPU array (wp.synchronize() was called
            # by the caller before this). np.ascontiguousarray() would make a redundant
            # full copy of the entire sonar grid — skip it and call tobytes() directly.
            sonar_data_np = sonar_data.numpy()  # shape: (range_bins, azimuth_bins)

            # Create ROS2 Image message
            msg = Image()
            sim_time_seconds = timeline.get_timeline_interface().get_current_time()
            msg.header.stamp = rclpy.time.Time(seconds=int(sim_time_seconds)).to_msg()
            msg.header.frame_id = frame_id
            msg.height = sonar_data_np.shape[0]
            msg.width = sonar_data_np.shape[1]
            msg.encoding = '32FC1'
            msg.is_bigendian = False
            msg.step = msg.width * 4  # 4 bytes per float32
            msg.data = sonar_data_np.tobytes()

            # Publish the message
            self._sonar_pub.publish(msg)

        except Exception as e:
            print(f'[{self._name}] Failed to publish sonar data: {e}')

    def make_sonar_data(self, 
                        binning_method: str = "sum", 
                        normalizing_method: str = "range",
                        query_prop: str ='reflectivity', # Do not modify this if not developing the sensor.
                        attenuation: float = 0.1, # Control the attentuation along distance when computing attenuation
                        gau_noise_param: float = 0.2, # multiplicative noise coefficient 
                        ray_noise_param: float = 0.05, # additive noise parameter
                        intensity_offset: float = 0.0, # offset intensity after normalization
                        intensity_gain: float = 1.0, # scale intensity after normalization
                        central_peak: float = 2, # control the strength of the streak
                        central_std: float = 0.001, # control the spread of the streak
                        ):
        """Process raw scan data into a sonar image with configurable parameters.

        Args:
            binning_method (str): "sum" or "mean" for intensity accumulation
                                Remember to adjust your noise scale accordingly after changing this.
            normalizing_method (str): "all" (global max) or "range" (per-range max)
                                Remember to adjust your noise scale accordingly after changing this.
            query_prop (str): Material property to query (default 'reflectivity')
                            Don't modify this if not for development.
            attenuation (float): Distance attenuation coefficient (0-1)
            gau_noise_param (float): Gaussian noise multiplier
            ray_noise_param (float): Rayleigh noise scale factor
            intensity_offset (float): Post-normalization intensity offset
            intensity_gain (float): Post-normalization intensity multiplier
            central_peak (float): Central beam streak intensity
            central_std (float): Central beam streak width
    
        """



        def make_indexToProp_array(idToLabels: dict, query_property: str):
            # A utility function helps to convert idToLabels into indexToProp array
            # This manipulation facilitates warp computation framework
            # indexToProp is an 1-dim array where the values associated with the query property 
            # are placed at the index corresponding to the key
            # First two entry are always zero because {'0': {'class': 'BACKGROUND'}, '1': {'class': 'UNLABELLED'}}
            # eg: indexToProp = [0, 0, 0.1, 1 .....] 
            max_id = max(idToLabels.keys(), default=-1)
            indexToProp_array = np.ones((int(max_id)+1,))
            for id in idToLabels.keys():
                for property in idToLabels.get(id):
                    if property == query_property:
                        indexToProp_array[int(id)] = idToLabels.get(id).get(property)
            return indexToProp_array

        if self.scan():
            num_points = self.scan_data['pcl'].shape[0]
            # Rebuild the GPU reflectivity LUT only when the label set changes.
            # In typical runs labels are static after warm-up, so this avoids
            # a Python loop + wp.array GPU upload on every single frame.
            current_labels = self.scan_data['idToLabels']
            if current_labels != self._cached_idToLabels:
                self._cached_idToLabels = current_labels
                self._cached_indexToRefl = wp.array(
                    make_indexToProp_array(idToLabels=current_labels, query_property=query_prop),
                    dtype=wp.float32
                )
            indexToRefl = self._cached_indexToRefl
            viewTransform = wp.mat44(self.scan_data['viewTransform'])
            # directly use warp array loaded on cuda
            pcl = self.scan_data['pcl']
            normals = self.scan_data['normals']
            semantics = self.scan_data['semantics']
        else:
            return

        # Use pre-allocated buffers (sliced to num_points) instead of wp.empty().
        # wp.empty() allocates a new GPU buffer every frame; the async free of the
        # previous frame's buffer races with the ROS2 numpy() memcpy_d2h, causing
        # CUDA error 700. Slicing a persistent buffer avoids all mid-run allocations.
        intensity = self._intensity_buf[:num_points]
        wp.launch(kernel=compute_intensity,
                  dim=num_points,
                  inputs=[
                      pcl,
                      normals,
                      viewTransform,
                      semantics,
                      indexToRefl,
                      attenuation,
                  ],
                  outputs=[
                      intensity
                  ]
                )
                
        # Transform pointcloud from world cooridates to sonar local
        pcl_local = self._pcl_local_buf[:num_points]
        pcl_spher = self._pcl_spher_buf[:num_points]
        wp.launch(kernel=world2local,
                  dim=num_points,
                  inputs=[
                      viewTransform,
                      pcl
                  ],
                    outputs=[
                      pcl_local,
                      pcl_spher
                    ]
                )
        
        # Collapse three dimensional intensity data to 2D
        # Simply sum intensity return and compute number of return that falls into the same bin
        self.bin_sum.zero_()
        self.bin_count.zero_()
        self.binned_intensity.zero_()

        
        wp.launch(kernel=bin_intensity,
                  dim=num_points,
                  inputs=[
                      pcl_spher,
                      intensity,
                      self.min_range,
                      self.min_azi,
                      self.range_res,
                      wp.radians(self.angular_res),
                  ],
                  outputs=[
                      self.bin_sum,
                      self.bin_count
                  ]
                  )
        
        # Process intensity data by either sum as it is or averaging
        if binning_method == "mean":
            wp.launch(
                kernel=average,
                dim=self.bin_sum.shape,
                inputs=[
                    self.bin_sum,
                    self.bin_count
                ],
                outputs=[
                    self.binned_intensity,
                ]
                )
        
        if binning_method == "sum":
            # Copy rather than reassign: reassigning self.binned_intensity to self.bin_sum
            # would break the pre-allocated buffer reference and cause a GPU alloc next frame.
            wp.copy(self.binned_intensity, self.bin_sum)


        self.range_dependent_ray_noise.zero_()
        self.gau_noise.zero_()
        self.sonar_map.zero_()

        # Calculate multiplicative gaussian noise
        
        wp.launch(
            kernel=normal_2d,
            dim=self.bin_sum.shape,
            inputs=[
                self.id,   # use frame num for RNG seed increment
                0.0,
                gau_noise_param
            ],
            outputs=[
                self.gau_noise
            ]
        )

        # Calculate additive rayleigh noise (range dependent and mimic central beam)

        wp.launch(
            kernel=range_dependent_rayleigh_2d,
            dim=self.bin_sum.shape,
            inputs=[
                self.id,   # use frame num for RNG seed increment
                self.r,
                self.azi,
                self.max_range,
                ray_noise_param,
                central_peak,
                central_std,
            ],
            outputs=[
                self.range_dependent_ray_noise

            ]
        )

        
        
        # Normalizing intensity at each bin either by global maximum or rangewise maximum
        # Compute global maximum
        if normalizing_method == "all":
            self._maximum_all.zero_()   # reset pre-allocated buffer instead of allocating
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=all_max,
                inputs=[
                    self.binned_intensity,
                ],
                outputs=[
                    self._maximum_all
                ]
            )
            
            # Apply noise, normalize by global maximum, and convert (r, azi) to (x,y) for plotting
            wp.launch(
                  kernel=make_sonar_map_all,
                  dim=self.sonar_map.shape,
                  inputs=[
                      self.r,
                      self.azi,
                      self.binned_intensity,
                      self._maximum_all,
                      self.gau_noise,
                      self.range_dependent_ray_noise,
                      intensity_offset,
                      intensity_gain
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
            
        if normalizing_method == "range":
            self._maximum_range.zero_()  # reset pre-allocated buffer instead of allocating
            wp.launch(
                dim=self.bin_sum.shape,
                kernel=range_max,
                inputs=[
                    self.binned_intensity,
                ],
                outputs=[
                    self._maximum_range
                ]
            )
            # Apply noise, normalize by range maximum, and convert (r, azi) to (x,y) for plotting
            wp.launch(
                  kernel=make_sonar_map_range,
                  dim=self.sonar_map.shape,
                  inputs=[
                      self.r,
                      self.azi, 
                      self.binned_intensity,
                      self._maximum_range,
                      self.gau_noise,
                      self.range_dependent_ray_noise,
                      intensity_offset,
                      intensity_gain
                  ],
                  outputs=[
                      self.sonar_map
                  ]
                  )
        
        
        # Write data to the dir
        if self.writing:
            # self.backend.schedule(write_np, f"intensity_{self.id}.npy", data=intensity)
            # self.backend.schedule(write_np, f'pcl_local_{self.id}.npy', data=pcl_local)
            self.backend.schedule(write_np, f'sonar_data_{self.id}.npy', data=self.sonar_map)
            print(f"[{self._name}] [{self.id}] Writing sonar data to {self.backend.output_dir}")
        
        if self._viewport:
            self._sonar_provider.set_bytes_data_from_gpu(self.make_sonar_image().ptr, 
                                                    [self.sonar_map.shape[1], self.sonar_map.shape[0]])
            # self.backend.schedule(write_image, f'sonar_{self.id}.png', data = self.make_sonar_image())        
            
        self.id += 1

        # ROS2 publishing
        if self._enable_ros2_pub:
            # Synchronize the CUDA device before calling numpy() inside the publisher.
            # numpy() triggers a synchronous memcpy_d2h. Without this sync, pending
            # async Warp kernel launches / frees can still be in-flight on the GPU,
            # and the concurrent memcpy corrupts the CUDA context (error 700).
            wp.synchronize()
            self._ros2_publish_sonar_image(self.binned_intensity)
            self.publish_pitch()

    

    def make_sonar_image(self):
        """Convert processed sonar data to a viewable grayscale image.
    
        Returns:
            wp.array: GPU array containing the sonar image (RGBA format)
    
        Note:
            - Used internally for viewport display
            - Image dimensions match the sonar's polar binning resolution
        """
        self.sonar_image.zero_()
        wp.launch(
            dim=self.sonar_map.shape,
            kernel=make_sonar_image,
            inputs=[
                self.sonar_map
            ],
            outputs=[
                self.sonar_image
            ]
        )
        return self.sonar_image
    

    def make_sonar_viewport(self):
        """Create an interactive viewport window for real-time sonar visualization.
    
        Note:
            - Displays live sonar images when simulation is running
            - Includes range and azimuth tick marks
            - Window size is fixed at 800x800 pixels
        """
        self.wrapped_ui_elements = []

        range_tick_num = 10
        range_tick = np.round(np.linspace(self.min_range, self.max_range, range_tick_num), 2)

        azi_tick_num = 10
        azi_tick = np.round(np.linspace(90-self.hori_fov/2, 90+self.hori_fov/2, azi_tick_num))
        self._sonar_provider = ui.ByteImageProvider()
        self._window = ui.Window(self._name, width=800, height=800, visible=True)
        
        with self._window.frame:
            with ui.ZStack(height=720, width = 720):
                ui.Rectangle(widthstyle={"background_color": 0xFF000000})
                ui.Label('Run the scenario for image to be received',
                         style={'font_size': 55,'alignment': ui.Alignment.CENTER},
                         word_wrap=True)
                sonar_image_provider = ui.ImageWithProvider(self._sonar_provider, 
                                    style={"width": 720, 
                                        "height": 720, 
                                        "fill_policy" : ui.FillPolicy.STRETCH,
                                        'alignment': ui.Alignment.CENTER})
                
                # ui.Line(alignment=ui.Alignment.LEFT,
                #         style={'border_width': 2,
                #                 'color':ui.color.white })
                # with ui.VGrid(row_height = 720/(range_tick_num-1)):
                #     for i in range(range_tick_num-1):
                #         with ui.ZStack():
                #             ui.Rectangle(style={'border_color': ui.color.white, 'background_color': ui.color.transparent,'border_width': 0.05, 'margin': 0})
                #             ui.Label(str(range_tick[i]) + ' m',style={'font_size': 15,'alignment': ui.Alignment.LEFT, 'margin':2})
                # with ui.HGrid(column_width = 720/(azi_tick_num-1), direction=ui.Direction.RIGHT_TO_LEFT):
                #     for i in range(azi_tick_num-1):
                #         with ui.ZStack():
                #             ui.Rectangle(style={'border_color': ui.color.white, 'background_color': ui.color.transparent,'border_width': 0.05, 'margin': 0})
                #             ui.Label(str(azi_tick[i]) + "°",style={'font_size': 15,'alignment': ui.Alignment.RIGHT, 'margin':2})                           
                # ui.Label(str(range_tick[-1]) +" m", style={'font_size': 15, "alignment":ui.Alignment.LEFT_BOTTOM, 'margin':2})
        
        self.wrapped_ui_elements.append(sonar_image_provider)
        self.wrapped_ui_elements.append(self._sonar_provider)
        self.wrapped_ui_elements.append(self._window)

    def get_range(self) -> list[float]:
        """Get the configured operating range of the sonar.
    
        Returns:
            list[float]: [min_range, max_range] in meters
        """
        return [self.min_range, self.max_range]
    
    def get_fov(self) -> list[float]:
        """Get the configured field of view angles.
    
        Returns:
            list[float]: [horizontal_fov, vertical_fov] in degrees
        """
        return [self.hori_fov, self.vert_fov]
    

    
    def close(self):
        """Clean up resources by detaching annotators and clearing caches.
    
        Note:
            - Required for proper shutdown when done using the sensor
            - Also closes viewport window if one was created
        """
        if self.pointcloud_annot:
            self.pointcloud_annot.detach(self._render_product_path)
            rep.AnnotatorCache.clear(self.pointcloud_annot)
            self.pointcloud_annot = None

        if self.cameraParams_annot:
            self.cameraParams_annot.detach(self._render_product_path)
            rep.AnnotatorCache.clear(self.cameraParams_annot)
            self.cameraParams_annot = None
            
        if self.semanticSeg_annot:
            self.semanticSeg_annot.detach(self._render_product_path)
            rep.AnnotatorCache.clear(self.semanticSeg_annot)
            self.semanticSeg_annot = None


        print(f'[{self._name}] Annotator detached. AnnotatorCache cleaned.')

        if self._viewport:
            self.ui_destroy()

        if hasattr(self, '_ros2_sonar_node') and self._ros2_sonar_node is not None:
            self._ros2_sonar_node.destroy_node()
            self._ros2_sonar_node = None


    def ui_destroy(self):
        """Explicitly destroy viewport UI elements.
    
        Note:
            - Called automatically by close()
            - Only needed if manually managing UI lifecycle
        """
        for elem in self.wrapped_ui_elements:
            elem.destroy()

    def add_debug_lines(self):
        """Visualize Imaging Sonar FOV in the viewport using debug drawing.
        
        Creates 4 dummy LightBeamSensors at the corners of the FOV and 
        an action graph that continuously draws the beam paths.
        """
        # Calculate orientations for the 4 corners of the frustum
        # Assuming camera convention: -Z forward, +Y up, +X right
        # Corners:
        # 1. Top-Left: (+vert/2, +hori/2) (Rotation order matters, we'll try ZYX or similar)
        # Using simple Euler angles for approximation
        
        h_half = self.hori_fov / 2.0
        v_half = self.vert_fov / 2.0
        
        # Define the 4 corners (Y-axis rotation is Yaw/Horizontal, X-axis rotation is Pitch/Vertical)
        # Note: Signs depend on axis definition. 
        # Rot Y (+): turns Z towards X (Right if -Z is Fwd? No, Z to X is +Y rot. (0,0,1)->(1,0,0))
        # If -Z is forward. Rot Y(+90) -> -X. So +Y rot is Left. -Y rot is Right.
        # Rot X (+): turns Y towards Z. (0,1,0)->(0,0,1). 
        # If -Z is forward. Rot X(+90) -> Old Y becomes Z(Back). Old -Z(Fwd) becomes Y(Up).
        # So +X rot is Pitch Up.
        
        # Corners (Pitch, Yaw, Roll):
        # TL: (+v, +h) -> Up, Left
        # TR: (+v, -h) -> Up, Right
        # BL: (-v, +h) -> Down, Left
        # BR: (-v, -h) -> Down, Right
        
        orients_euler = np.array([
            [v_half, h_half, 0.0],  # TL
            [v_half, -h_half, 0.0], # TR
            [-v_half, h_half, 0.0], # BL
            [-v_half, -h_half, 0.0] # BR
        ])
        
        self._debug_beam_paths = []
        
        for i in range(4):
            path = self.prim_path + f"/debug_beam_{i}"
            self._debug_beam_paths.append(path)
            
            # Create dummy beam sensor
            result, _ = omni.kit.commands.execute(
                "IsaacSensorCreateLightBeamSensor",
                path=path,
                min_range=self.min_range,
                max_range=self.max_range,
                forward_axis=Gf.Vec3d(0, 0, -1),
                num_rays=1,
            )
            
            if result:
                # Set orientation
                quat = euler_angles_to_quat(orients_euler[i], degrees=True)
                SingleXFormPrim(prim_path=path).set_local_pose(orientation=quat)
            else:
                carb.log_error(f"[{self._name}] Failed to create debug beam {i}")

        # Create Action Graph for visualization
        try:
            (action_graph, new_nodes, _, _) = og.Controller.edit(
                {"graph_path": "/debugLinesSonar", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("IsaacReadLightBeam0", "isaacsim.sensors.physx.IsaacReadLightBeam"),
                        ("IsaacReadLightBeam1", "isaacsim.sensors.physx.IsaacReadLightBeam"),
                        ("IsaacReadLightBeam2", "isaacsim.sensors.physx.IsaacReadLightBeam"),
                        ("IsaacReadLightBeam3", "isaacsim.sensors.physx.IsaacReadLightBeam"),
                        ("DebugDrawRayCast0", "isaacsim.util.debug_draw.DebugDrawRayCast"),
                        ("DebugDrawRayCast1", "isaacsim.util.debug_draw.DebugDrawRayCast"),
                        ("DebugDrawRayCast2", "isaacsim.util.debug_draw.DebugDrawRayCast"),
                        ("DebugDrawRayCast3", "isaacsim.util.debug_draw.DebugDrawRayCast"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("IsaacReadLightBeam0.inputs:lightbeamPrim", self._debug_beam_paths[0]),
                        ("IsaacReadLightBeam1.inputs:lightbeamPrim", self._debug_beam_paths[1]),
                        ("IsaacReadLightBeam2.inputs:lightbeamPrim", self._debug_beam_paths[2]),
                        ("IsaacReadLightBeam3.inputs:lightbeamPrim", self._debug_beam_paths[3]),
                        # Set color to red for visibility
                        ("DebugDrawRayCast0.inputs:color", [1, 0, 0, 1]),
                        ("DebugDrawRayCast1.inputs:color", [1, 0, 0, 1]),
                        ("DebugDrawRayCast2.inputs:color", [1, 0, 0, 1]),
                        ("DebugDrawRayCast3.inputs:color", [1, 0, 0, 1]),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam0.inputs:execIn"),
                        ("IsaacReadLightBeam0.outputs:execOut", "DebugDrawRayCast0.inputs:exec"),
                        ("IsaacReadLightBeam0.outputs:beamOrigins", "DebugDrawRayCast0.inputs:beamOrigins"),
                        ("IsaacReadLightBeam0.outputs:beamEndPoints", "DebugDrawRayCast0.inputs:beamEndPoints"),
                        ("IsaacReadLightBeam0.outputs:numRays", "DebugDrawRayCast0.inputs:numRays"),

                        ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam1.inputs:execIn"),
                        ("IsaacReadLightBeam1.outputs:execOut", "DebugDrawRayCast1.inputs:exec"),
                        ("IsaacReadLightBeam1.outputs:beamOrigins", "DebugDrawRayCast1.inputs:beamOrigins"),
                        ("IsaacReadLightBeam1.outputs:beamEndPoints", "DebugDrawRayCast1.inputs:beamEndPoints"),
                        ("IsaacReadLightBeam1.outputs:numRays", "DebugDrawRayCast1.inputs:numRays"),

                        ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam2.inputs:execIn"),
                        ("IsaacReadLightBeam2.outputs:execOut", "DebugDrawRayCast2.inputs:exec"),
                        ("IsaacReadLightBeam2.outputs:beamOrigins", "DebugDrawRayCast2.inputs:beamOrigins"),
                        ("IsaacReadLightBeam2.outputs:beamEndPoints", "DebugDrawRayCast2.inputs:beamEndPoints"),
                        ("IsaacReadLightBeam2.outputs:numRays", "DebugDrawRayCast2.inputs:numRays"),

                        ("OnPlaybackTick.outputs:tick", "IsaacReadLightBeam3.inputs:execIn"),
                        ("IsaacReadLightBeam3.outputs:execOut", "DebugDrawRayCast3.inputs:exec"),
                        ("IsaacReadLightBeam3.outputs:beamOrigins", "DebugDrawRayCast3.inputs:beamOrigins"),
                        ("IsaacReadLightBeam3.outputs:beamEndPoints", "DebugDrawRayCast3.inputs:beamEndPoints"),
                        ("IsaacReadLightBeam3.outputs:numRays", "DebugDrawRayCast3.inputs:numRays"),
                    ],
                },
            )
        except Exception as e:
            carb.log_error(f"[{self._name}] Failed to create debug graph: {e}")