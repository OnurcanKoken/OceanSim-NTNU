# Omniverse import
import numpy as np
import os
import traceback
import omni.timeline
import omni.ui as ui
from omni.usd import StageEventType
from pxr import PhysxSchema
from isaacsim.oceansim.utils.assets_utils import get_imu_config_path, get_waypoints_default_path
import carb

# Isaac sim import
from isaacsim.core.prims import SingleRigidPrim, SingleGeometryPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_update_semantics
from isaacsim.gui.components import CollapsableFrame, StateButton, get_style, setup_ui_headers, CheckBox, combo_cb_xyz_plot_builder, combo_cb_plot_builder, dropdown_builder, str_builder
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.examples.extension.core_connectors import LoadButton, ResetButton
from isaacsim.core.utils.extensions import get_extension_path

# Custom import
from .scenario import MHL_Sensor_Example_Scenario
from .global_variables import EXTENSION_DESCRIPTION, EXTENSION_TITLE, EXTENSION_LINK
from isaacsim.oceansim.utils.assets_utils import (
    get_scene_path, load_config, get_assets_root,
    list_dynamics_configs, load_dynamics_config, get_dynamics_config_path
)

class UIBuilder():
    def __init__(self):

        self._ext_id = omni.kit.app.get_app().get_extension_manager().get_extension_id_by_module(__name__)
        self._file_path = os.path.abspath(__file__)
        self._title = EXTENSION_TITLE
        self._doc_link =  EXTENSION_LINK
        self._overview = EXTENSION_DESCRIPTION
        self._extension_path = get_extension_path(self._ext_id)
        
        self._ctrl_mode = 'Manual control' #"ROS control" 
        self._waypoints_path = get_waypoints_default_path()
        self._data_collection_path = ""
        
        # Calculate IMU config path relative to this file
        self._imu_config_path = get_imu_config_path()
        # Get access to the timeline to control stop/pause/play programmatically
        self._timeline = omni.timeline.get_timeline_interface()

        # UI frames created
        self.frames = []
        # UI elements created using a UIElementWrapper instance
        self.wrapped_ui_elements = []

        # Run initialization for the provided example
        self._on_init()

        self.physics_freq = 200.0 # Physics frequency in the simulator [Hz]

        self._data_collection_mode = True

    ###################################################################################
    #           The Functions Below Are Called Automatically By extension.py
    ###################################################################################

    def on_menu_callback(self):
        """Callback for when the UI is opened from the toolbar.
        This is called directly after build_ui().
        """
        pass

    def on_timeline_event(self, event):
        """Callback for Timeline events (Play, Pause, Stop)

        Args:
            event (omni.timeline.TimelineEventType): Event Type
        """
        if event.type == int(omni.timeline.TimelineEventType.STOP):
            # When the user hits the stop button through the UI, they will inevitably discover edge cases where things break
            # For complete robustness, the user should resolve those edge cases here
            # In general, for extensions based off this template, there is no value to having the user click the play/stop
            # button instead of using the Load/Reset/Run buttons provided.
            self._scenario_state_btn.reset()
            self._scenario_state_btn.enabled = False

    def on_physics_step(self, step: float):
        """Callback for Physics Step.
        Physics steps only occur when the timeline is playing

        Args:
            step (float): Size of physics step
        """
        pass

    def on_stage_event(self, event):
        """Callback for Stage Events

        Args:
            event (omni.usd.StageEventType): Event Type
        """
        if event.type == int(StageEventType.OPENED):
            # If the user opens a new stage, the extension should completely reset
            self._reset_extension()

    def cleanup(self):
        """
        Called when the stage is closed or the extension is hot reloaded.
        Perform any necessary cleanup such as removing active callback functions
        Buttons imported from omni.isaac.ui.element_wrappers implement a cleanup function that should be called
        """
        if self._scenario:
            try:
                self._scenario.destroy()
            except Exception as e:
                print(f"Error destroying scenario: {e}")
                
        self._DVL_event_sub = None
        self._baro_event_sub = None
        self._IMU_event_sub_gyro = None  # <--- CORRECT NAME
        self._IMU_event_sub_accel = None # <--- CORRECT NAME
        for ui_elem in self.wrapped_ui_elements:
            ui_elem.cleanup()
        for frame in self.frames:
            frame.cleanup()
            
    def build_ui(self):
        """
        Build a custom UI tool to run your extension.
        This function will be called any time the UI window is closed and reopened.
        """

        setup_ui_headers(
            ext_id=self._ext_id, 
            file_path=self._file_path, 
            title=self._title, 
            doc_link=self._doc_link, 
            overview=self._overview, 
            info_collapsed=False
        )

        self._build_sensor_frame()
        self._build_world_controls_frame()
        self._build_run_scenario_frame()
        self._build_hidden_frames()

    def _build_sensor_frame(self):
        sensor_choosing_frame = CollapsableFrame('Sensors', collapsed=False)
        self.frames.append(sensor_choosing_frame)
        with sensor_choosing_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                sonar_check_box = CheckBox(
                    "Imaging Sonar",
                    default_value=False,
                    tooltip=" Click this checkbox to activate imaging sonar",
                    on_click_fn=self._on_sonar_checkbox_click_fn,
                )
                self._use_sonar = False
                self.wrapped_ui_elements.append(sonar_check_box)
                camera_check_box = CheckBox(
                    "Underwater Camera",
                    default_value=False,
                    tooltip=" Click this checkbox to activate underwater camera",
                    on_click_fn=self._on_camera_checkbox_click_fn,
                )
                self._use_camera = False
                self.wrapped_ui_elements.append(camera_check_box)

                stereo_cam_check_box = CheckBox(
                    "Stereo Downward Cameras",
                    default_value=False,
                    tooltip=" Click this checkbox to activate stereo downward cameras (0.12m baseline)",
                    on_click_fn=self._on_stereo_camera_checkbox_click_fn,
                )
                self._use_stereo_camera = False
                self.wrapped_ui_elements.append(stereo_cam_check_box)

                self._uw_yaml_path_field = str_builder(
                    label='Path to UW Config',
                    default_val="",
                    tooltip='Select the YAML file for underwater camera config (optional)',
                    use_folder_picker=True,
                    folder_button_title="Select YAML",
                    folder_dialog_title='Select UW Camera YAML Config'
                )
                self._uw_yaml_path_field.add_value_changed_fn(self._on_uw_yaml_path_changed_fn)

                DVL_check_box = CheckBox(
                    'DVL',
                    default_value=False,
                    tooltip=" Click this checkbox to activate DVL",
                    on_click_fn=self._on_DVL_checkbox_click_fn
                )
                self._use_DVL = False
                self.wrapped_ui_elements.append(DVL_check_box)

                baro_check_box = CheckBox(
                    "Barometer",
                    default_value=False,
                    tooltip='Click this checkbox to activate barometer',
                    on_click_fn=self._on_baro_checkbox_click_fn
                ) 
                self._use_baro = False
                self.wrapped_ui_elements.append(baro_check_box)

                self.accel_check_box = CheckBox(
                    "Accelerometer",
                    default_value=False,
                    tooltip='Click this checkbox to activate Accelerometer',   
                    on_click_fn=self._on_Accel_checkbox_click_fn
                )
                self._use_IMU = False
                self.wrapped_ui_elements.append(self.accel_check_box)
                self.gyro_check_box = CheckBox(
                    "Gyroscope",
                    default_value=False,
                    tooltip='Click this checkbox to activate Gyroscope',
                    on_click_fn=self._on_Gyro_checkbox_click_fn
                )
                self._use_IMU = False
                self.wrapped_ui_elements.append(self.gyro_check_box)
                #data collection mode checkbox
                self._data_collection_mode = False
                self.data_collection_check_box = CheckBox(
                    "Data Collection Mode",
                    default_value=False,
                    tooltip=' Click this checkbox to activate data collection mode',
                    on_click_fn=self._on_data_collection_cb_click_fn
                )

    def _build_world_controls_frame(self):
        world_controls_frame = CollapsableFrame("World Controls", collapsed=False)
        self.frames.append(world_controls_frame)
        with world_controls_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                
                # self._build_USD_filepicker()
                self._USD_path_field = str_builder(
                    label='Path to USD',
                    default_val="",
                    tooltip='Select the USD file for the scene',
                    use_folder_picker=True,
                    folder_button_title="Select USD",
                    folder_dialog_title='Select the USD scene to test')
                
                self._ctrl_mode_model = dropdown_builder(
                    label='Control Mode',
                    default_val=3,
                    items=['No control', 'Straight line', 'Waypoints', 'Manual control', 'ROS control', 'ROS + Manual control'],
                    tooltip='Select preferred control mode',
                    on_clicked_fn=self._on_ctrl_mode_dropdown_clicked
                )

                self._load_btn = LoadButton(
                    "Load Button", "LOAD", setup_scene_fn=self._setup_scene, setup_post_load_fn=self._setup_scenario
                )
                self._load_btn.set_world_settings(physics_dt=1/self.physics_freq, rendering_dt=1/25.0)
                self.wrapped_ui_elements.append(self._load_btn)

                self._reset_btn = ResetButton(
                    "Reset Button", "RESET", pre_reset_fn=None, post_reset_fn=self._on_post_reset_btn
                )
                self._reset_btn.enabled = False
                self.wrapped_ui_elements.append(self._reset_btn)

    def _build_run_scenario_frame(self):
        run_scenario_frame = CollapsableFrame("Run Scenario", collapsed=False)
        self.frames.append(run_scenario_frame)
        with run_scenario_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._scenario_state_btn = StateButton(
                    "Run Scenario",
                    "RUN",
                    "STOP",
                    on_a_click_fn=self._on_run_scenario_a_text,
                    on_b_click_fn=self._on_run_scenario_b_text,
                    physics_callback_fn=self._update_scenario,
                )
                self._scenario_state_btn.enabled = False
                self.wrapped_ui_elements.append(self._scenario_state_btn)

    def _build_hidden_frames(self):
        self.sensor_reading_frame = CollapsableFrame('Sensor Reading', collapsed=False, visible=False)
        self.frames.append(self.sensor_reading_frame)
        self.waypoints_frame = CollapsableFrame('Waypoints',collapsed=False, visible=False)
        self.frames.append(self.waypoints_frame)
        self.ros2_control_frame = CollapsableFrame('ROS2 Control Mode Setting', collapsed=False, visible=False)
        self.frames.append(self.ros2_control_frame)
        self.data_collection_frame = CollapsableFrame('Data Collection Settings', collapsed=False, visible=False)
        self.frames.append(self.data_collection_frame)
        with self.data_collection_frame:
            # Pre-build the data collection path field here
            self._data_save_path_field = str_builder(
                label='Path to save data',
                default_val="",
                tooltip='Select the folder to save the collected sensor data',
                use_folder_picker=True,
                folder_button_title="Select Folder",
                folder_dialog_title='Select the folder to save the collected sensor data'
            )
            self._data_save_path_field.add_value_changed_fn(self._on_data_save_path_changed_fn)

    ######################################################################################
    # Functions Below This Point Related to Scene Setup (USD\PhysX..)
    ######################################################################################

    def _on_init(self):

        # Dynamics config — robot parameters are loaded from the selected YAML
        self._dynamics_configs = list_dynamics_configs()
        self._dynamics_config_name = self._dynamics_configs[0] if self._dynamics_configs else "benzon"
        self._load_robot_params_from_dynamics()
        self._uw_yaml_path = ""

        # Sensor
        self._sonar = None
        self._sonar_trans = np.array([0.3,0.0, 0.3])
        self._sonar = None
        self._sonar_trans = np.array([0.3,0.0, 0.3])
        self._cams = []
        self._cam_trans = np.array([0.3,0.0, 0.1])
        self._cam_focal_length = 21
        self._DVL = None
        self._DVL_trans = np.array([0,0,-0.1])
        self._baro = None
        self._water_surface = 1.43389 # Arbitrary
        self._IMU = None
        self._IMU_trans = np.array([0, 0, 0])
        self._IMU_orient = np.array([1, 0, 0, 0])


        
        # Scenario
        self._scenario = MHL_Sensor_Example_Scenario()
        


    def _load_robot_params_from_dynamics(self):
        """Load robot PhysX parameters from the selected dynamics config YAML."""
        try:
            cfg = load_dynamics_config(self._dynamics_config_name)
            vehicle = cfg.get('vehicle', {})
            robot = cfg.get('robot', {})
            self._rob_mass = vehicle.get('mass', 13.5)
            self._rob_linear_damping = robot.get('physx_linear_damping', 0.5)
            self._rob_angular_damping = robot.get('physx_angular_damping', 0.5)
            self._rob_disable_gravity = robot.get('disable_gravity', False)
            self._dynamics_config_path = get_dynamics_config_path(self._dynamics_config_name)
            print(f"[UIBuilder] Loaded robot params from '{self._dynamics_config_name}': "
                  f"mass={self._rob_mass}kg, lin_damp={self._rob_linear_damping}, "
                  f"ang_damp={self._rob_angular_damping}, gravity_disabled={self._rob_disable_gravity}")
        except Exception as e:
            print(f"[UIBuilder] Failed to load dynamics config '{self._dynamics_config_name}': {e}")
            self._rob_mass = 13.5
            self._rob_linear_damping = 0.5
            self._rob_angular_damping = 0.5
            self._rob_disable_gravity = False
            self._dynamics_config_path = None

    def _setup_scene(self):
        """
        This function is attached to the Load Button as the setup_scene_fn callback.
        On pressing the Load Button, a new instance of World() is created and then this function is called.
        The user should now load their assets onto the stage and add them to the World Scene.
        """
        create_new_stage()
        self._setup_environment()
        self._setup_robot()
        self._init_sensors()

    def _setup_environment(self):
        if self._USD_path_field.get_value_as_string() != "":
            scene_prim_path = '/World/scene'
            add_reference_to_stage(usd_path=self._USD_path_field.get_value_as_string(), prim_path=scene_prim_path)
            print('User USD scene is loaded.')
        else:
            print('USD path is empty. Default to example scene')

            # Load environment elements from config
            config = load_config()
            environment = config.get("paths", {}).get("environment", [])
            print(f"[DEBUG] Environment config: {environment}")
            
            if not environment:
                print("[WARN] Environment list is empty or missing in config!")

            for item in environment:
                usd_path = os.path.join(get_assets_root(), item["usd_path"])
                prim_path = item["prim_path"]
                print(f"[DEBUG] Loading: {item['name']} | USD: {usd_path} | Prim: {prim_path}")
                
                # Add reference
                add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
                
                # Transforms
                if "translation" in item:
                    # We need to apply translation to the prim
                    # But add_reference returns the primitive if we are lucky? 
                    # Actually add_reference_to_stage returns the prim.
                    pass 

                # Collision
                if item.get("collision", False):
                     SingleGeometryPrim(prim_path=prim_path, collision=True)
                     
                prim = get_prim_at_path(prim_path)

                if "collision_approximation" in item:
                     # We need SingleGeometryPrim wrapper usually
                     geo_prim = SingleGeometryPrim(prim_path=prim_path)
                     geo_prim.set_collision_approximation(item["collision_approximation"])

                # Rigid Body
                if item.get("rigid_body", False):
                    trans = np.array(item.get("translation", [0,0,0]))
                    orient = None
                    if "orientation" in item:
                        orient = euler_angles_to_quat(np.array(item["orientation"]), degrees=True)
                        
                    scale = None
                    if "scale" in item:
                        scale = np.array(item["scale"])
                        
                    SingleRigidPrim(prim_path=prim_path,
                                    translation=trans,
                                    orientation=orient,
                                    scale=scale,
                                    mass=item.get("mass", None))
                else:
                    # If not rigid body, but has transform
                     if "translation" in item or "orientation" in item:
                        # Use XformPrim or just set attributes?
                        # SingleRigidPrim is easy wrapper but adds RigidBodyAPI.
                        # For static items, we might just want to set xform.
                        # But in Isaac Core, XformPrim can be used.
                        from isaacsim.core.prims import XformPrim
                        orient = None
                        if "orientation" in item:
                            orient = euler_angles_to_quat(np.array(item["orientation"]), degrees=True)
                        trans = None
                        if "translation" in item:
                            trans = np.array(item["translation"])
                        scale = None
                        if "scale" in item:
                            scale = np.array(item["scale"])
                            
                        xp = XformPrim(prim_path=prim_path, translation=trans, orientation=orient, scale=scale)

                # Semantics
                if "semantics" in item:
                    for sem in item["semantics"]:
                        # path relative to prim_path
                        target_path = f"{prim_path}/{sem['path']}"
                        prim = get_prim_at_path(target_path)
                        if prim.IsValid():
                            add_update_semantics(prim=prim,
                                                type_label=sem['type'],
                                                semantic_label=sem['value'])
                        else:
                            print(f"[WARN] Could not find prim at {target_path} to apply semantics.")

    def _setup_robot(self):
        # add bluerov robot as reference
        robot_prim_path = "/World/rob"
        robot_usd_path = get_scene_path("robot")
        self._rob = add_reference_to_stage(usd_path=robot_usd_path, prim_path=robot_prim_path)
        # Gravity setting from dynamics config — hydrodynamic model provides buoyancy
        rob_rigidBody_API = PhysxSchema.PhysxRigidBodyAPI.Apply(get_prim_at_path(robot_prim_path))
        rob_rigidBody_API.CreateDisableGravityAttr(self._rob_disable_gravity)
        # Set damping of the robot
        rob_rigidBody_API.GetLinearDampingAttr().Set(self._rob_linear_damping)
        rob_rigidBody_API.GetAngularDampingAttr().Set(self._rob_angular_damping)
        # Set the mass for the robot to suppress a warning from inertia autocomputation
        rob_collider_prim = SingleGeometryPrim(prim_path=robot_prim_path,
                                               collision=True)
        rob_collider_prim.set_collision_approximation('boundingCube')
        SingleRigidPrim(prim_path=robot_prim_path,
                        mass=self._rob_mass,
                        translation=np.array([-2.0, 0.0, -0.8]))

        set_camera_view(eye=np.array([5,0.6,0.4]), target=rob_collider_prim.get_world_pose()[0])

    def _init_sensors(self):
        robot_prim_path = "/World/rob"
        if self._use_sonar:
            print("[Debug] Initializing Sonar...")
            # Cleanup old sonar if it exists to prevent Replicator conflicts
            if self._sonar is not None:
                print("[Debug] Cleaning up old Sonar instance...")
                try:
                    self._sonar.close()
                except Exception as e:
                    print(f"[Debug] Error closing old Sonar: {e}")
                self._sonar = None

            from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor
            self._sonar = ImagingSonarSensor(prim_path=robot_prim_path + '/sonar',
                                            translation=self._sonar_trans,
                                            orientation=euler_angles_to_quat(np.array([0.0, 45, 0.0]),  degrees=True),
                                            range_res=0.005,
                                            angular_res=0.25,
                                            hori_res=4000
                                            )
            # self._sonar.add_debug_lines()
            
        from isaacsim.oceansim.sensors.UW_Camera import UW_Camera
        from omni.isaac.core import World
        
        # Helper to setup camera
        def setup_camera(prim_path, name, translation, orientation=None, resolution=[1920, 1080]):
            cam = UW_Camera(prim_path=prim_path,
                            name=name,
                            resolution=resolution,
                            translation=translation,
                            orientation=orientation)
            cam.set_focal_length(0.1 * self._cam_focal_length)
            cam.set_clipping_range(0.1, 100)
            
            world = World.instance()
            if world is None:
                carb.log_warn(f"World instance is None. Camera {name} frequency default to 60Hz.")
                cam.set_frequency(60)
            else:
                render_dt = world.get_rendering_dt()
                if render_dt is not None and render_dt > 0:
                    render_freq = 1.0 / render_dt
                    cam.set_frequency(int(render_freq))
                else:
                    cam.set_frequency(60)
            return cam

        if self._use_camera:
            self._cams.append(setup_camera(
                prim_path=robot_prim_path + '/UW_camera',
                name="UW_Camera",
                translation=self._cam_trans
            ))
            
        if self._use_stereo_camera:
            # 0.12m baseline, centered at Y=0 -> +/- 0.06
            # Downward looking: Rotation around Y axis by -90 degrees? 
            # Standard camera looks at -Z. 
            # Robot frame: X forward, Z up.
            # We want camera looking at -Z (down). 
            # If we rotate 90 deg around Y? 
            # Original: X right, Y up, Z forward (OpenGL)?? No, USD is -Z forward, Y up, X right.
            # Wait, let's use euler_angles_to_quat to be safe and consistent.
            # Downward usually means pitch -90.
            
            # Left Camera (Y = +0.06)
            self._cams.append(setup_camera(
                prim_path=robot_prim_path + '/down_cam_left',
                name="DownCamLeft",
                translation=np.array([-0.2, 0.06, -0.01]), # Adjust X/Z as needed
                orientation=euler_angles_to_quat(np.array([0.0, 90.0, 0.0]), degrees=True),
                resolution=[1280, 720] # Lower resolution for stereo? Or same? Keeping high for now or default
            ))
            
            # Right Camera (Y = -0.06)
            self._cams.append(setup_camera(
                prim_path=robot_prim_path + '/down_cam_right',
                name="DownCamRight",
                translation=np.array([-0.2, -0.06, -0.01]),
                orientation=euler_angles_to_quat(np.array([0.0, 90.0, 0.0]), degrees=True),
                resolution=[1280, 720]
            ))
            
        if self._use_DVL:
            from isaacsim.oceansim.sensors.DVLsensor import DVLsensor

            self._DVL = DVLsensor(max_range=10, freq=10)
            self._DVL.attachDVL(rigid_body_path=robot_prim_path,
                                translation=self._DVL_trans)
            # self._DVL.add_debug_lines()
            
        if self._use_baro:
            from isaacsim.oceansim.sensors.BarometerSensor import BarometerSensor

            self._baro = BarometerSensor(prim_path=robot_prim_path + '/Baro',
                                        water_surface_z=self._water_surface)
        if self._use_IMU:
            from isaacsim.oceansim.sensors.IMU import IMU

            self._IMU = IMU(prim_path=robot_prim_path + '/IMU',
                            translation=self._IMU_trans,
                            orientation=self._IMU_orient,
                            config_path=self._imu_config_path
                            )
            


    def _setup_scenario(self):
        """
        This function is attached to the Load Button as the setup_post_load_fn callback.
        The user may assume that their assets have been loaded by t setup_scene_fn callback, that
        their objects are properly initialized, and that the timeline is paused on timestep 0.
        """
        self._reset_scenario()
        self._add_extra_ui()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True
        self._reset_btn.enabled = True

    def _reset_scenario(self):
        self._scenario.teardown_scenario()
        self._scenario.setup_scenario(
            self._rob,
            self._sonar,
            self._cams,
            self._DVL,
            self._baro,
            self._IMU,
            self._ctrl_mode,
            self._data_collection_mode,
            data_collection_path=self._data_collection_path,
            uw_yaml_path=self._uw_yaml_path_field.get_value_as_string() if hasattr(self, '_uw_yaml_path_field') else "",
            dynamics_config_path=self._dynamics_config_path
            )
        self._scenario.setup_waypoints(
            waypoint_path=self._waypoints_path, 
            default_waypoint_path=self._extension_path + '/demo/demo_waypoints.txt'
            )

    def _on_post_reset_btn(self):
        """
        This function is attached to the Reset Button as the post_reset_fn callback.
        The user may assume that their objects are properly initialized, and that the timeline is paused on timestep 0.

        They may also assume that objects that were added to the World.Scene have been moved to their default positions.
        I.e. the cube prim will move back to the posiheirtion it was in when it was created in self._setup_scene().
        """
        self._reset_scenario()

        # UI management
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = True

    def _update_scenario(self, step: float):
        """This function is attached to the Run Scenario StateButton.
        This function was passed in as the physics_callback_fn argument.
        This means that when the a_text "RUN" is pressed, a subscription is made to call this function on every physics step.
        When the b_text "STOP" is pressed, the physics callback is removed.

        Args:
            step (float): The dt of the current physics step
        """
    
        self._scenario.update_scenario(step)

    def _on_run_scenario_a_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_a_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "RUN".

        This function simply plays the timeline, which means that physics steps will start happening.  After the world is loaded or reset,
        the timeline is paused, which means that no physics steps will occur until the user makes it play either programmatically or
        through the left-hand UI toolbar.
        """
        self._timeline.play()

    def _on_run_scenario_b_text(self):
        """
        This function is attached to the Run Scenario StateButton.
        This function was passed in as the on_b_click_fn argument.
        It is called when the StateButton is clicked while saying a_text "STOP"

        Pausing the timeline on b_text is not strictly necessary for this example to run.
        Clicking "STOP" will cancel the physics subscription that updates the scenario, which means that
        the robot will stop getting new commands and the cube will stop updating without needing to
        pause at all.  The reason that the timeline is paused here is to prevent the robot being carried
        forward by momentum for a few frames after the physics subscription is canceled.  Pausing here makes
        this example prettier, but if curious, the user should observe what happens when this line is removed.
        """
        self._timeline.pause()

    def _reset_extension(self):
        """This is called when the user opens a new stage from self.on_stage_event().
        All state should be reset.
        """
        self._on_init()
        self._reset_ui()

    def _reset_ui(self):
        self._scenario_state_btn.reset()
        self._scenario_state_btn.enabled = False
        self._reset_btn.enabled = False


    def _on_sonar_checkbox_click_fn(self, model):
        self._use_sonar = model
        print('Reload the scene for changes to take effect.')

    def _on_camera_checkbox_click_fn(self, model):
        self._use_camera = model
        print('Reload the scene for changes to take effect.')

    def _on_stereo_camera_checkbox_click_fn(self, model):
        self._use_stereo_camera = model
        print('Reload the scene for changes to take effect.')

    def _on_DVL_checkbox_click_fn(self, model):
        self._use_DVL = model
        print('Reload the scene for changes to take effect.')

    def _on_baro_checkbox_click_fn(self, model):
        self._use_baro = model
        print('Reload the scene for changes to take effect.')
    
    def _on_Accel_checkbox_click_fn(self, model):
        self._use_IMU = model or self._gyro_check_box.get_value_as_bool()
        print('Reload the scene for changes to take effect.')
    def _on_Gyro_checkbox_click_fn(self, model):
        self._use_IMU = model or self._accel_check_box.get_value_as_bool()

    def _on_uw_yaml_path_changed_fn(self, model):
        self._uw_yaml_path = model.get_value_as_string()
        print(f'UW YAML Path: {self._uw_yaml_path}')
        print('Reload the scene for changes to take effect.')
    
    def _on_manual_ctrl_cb_click_fn(self, model):
        self._manual_ctrl = model
        print('Reload the scene for changes to take effect.')
    def _on_data_collection_cb_click_fn(self, model):
        self._data_collection_mode = model
        self.data_collection_frame.visible = model
        # print('Reload the scene for changes to take effect.')

    def _on_ctrl_mode_dropdown_clicked(self, model):
        self._ctrl_mode = model
        print(f'Ctrl mode: {model}. Reload the scene for changes to take effect.')

   
    def _add_extra_ui(self):
        with self.sensor_reading_frame:
            with ui.VStack(spacing=5, height=0):                
                if self._use_DVL is True:
                    self._build_DVL_plot()
                    self.sensor_reading_frame.visible = True
                if self._use_baro is True:
                    self._build_baro_plot()
                    self.sensor_reading_frame.visible = True
                if self._use_IMU is True:
                    self._build_gyro_plot()
                    self._build_accel_plot()
                    self.sensor_reading_frame.visible = True
                if not self._use_baro and not self._use_DVL and not self._use_IMU:
                    self.sensor_reading_frame.visible = False 
        with self.data_collection_frame:
            # Move the UI creation to build_ui, here we just show/hide
            if self._data_collection_mode:
                self.data_collection_frame.visible = True
            else:
                self.data_collection_frame.visible = False
        with self.waypoints_frame:
            if self._ctrl_mode == 'Waypoints':
                with ui.VStack(style=get_style(), spacing=5):
                    self._build_waypoints_filepicker()
                self.waypoints_frame.visible = True
            else:
                self.waypoints_frame.visible = False
        with self.ros2_control_frame:
            if self._ctrl_mode == 'ROS control' or self._ctrl_mode == 'ROS + Manual control':
                # Build the ROS2 control UI
                self._build_ros2_control_ui()
                self.ros2_control_frame.visible = True
            else:
                self.ros2_control_frame.visible = False

    def _build_ros2_control_ui(self):
        """Build the ROS2 control UI elements"""
        with self.ros2_control_frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # ROS2 control mode dropdown
                self._ros2_control_mode_model = dropdown_builder(
                    label='ROS2 Control Mode',
                    default_val=0,
                    items=['velocity control', 'force control'],
                    tooltip='Select preferred ROS2 control mode',
                    on_clicked_fn=self._on_ros2_control_mode_dropdown_clicked
                )

    def _on_ros2_control_mode_dropdown_clicked(self, mode):
        self._scenario._ros2_control_mode = mode
        self._scenario._ros2_control_receiver._setup_ros2_control_mode(
                self._scenario._ros2_control_mode
            )
        print(f'ROS control mode switch to: {self._scenario._ros2_control_mode}.')

    def _build_waypoints_filepicker(self):
        self._waypoints_path_field = str_builder(
            label='Path to waypoints',
            default_val=self._waypoints_path,
            tooltip='Select the txt files containing the waypoint data',
            use_folder_picker=True,
            folder_button_title='Select txt',
            folder_dialog_title='Select the txt file containing the waypoint'
        )
        self._scenario.setup_waypoints(
            waypoint_path=self._waypoints_path, 
            default_waypoint_path=self._extension_path + '/demo/demo_waypoints.txt'
            )
        self._waypoints_path_field.add_value_changed_fn(self._on_waypoints_path_changed_fn)
        
        def on_random_click():
            self._waypoints_path_field.set_value("RANDOM")
            # Trigger change manually
            self._on_waypoints_path_changed_fn(self._waypoints_path_field)

        ui.Button("Generate Random Waypoints", clicked_fn=on_random_click, height=20, style={"margin": 5})

    def _on_waypoints_path_changed_fn(self, model):
        self._waypoints_path = model.get_value_as_string()
        self._scenario.setup_waypoints(
            waypoint_path=model.get_value_as_string(), 
            default_waypoint_path=self._extension_path + '/demo/demo_waypoints.txt'
            )
    def _on_data_save_path_changed_fn(self, model):
        # Just update the local path variable. We will pass it to scenario setup when LOAD is clicked.
        try:
            path = model.get_value_as_string()
            self._data_collection_path = path
            print('Data save path selected (will be used on Load): ', path)
        except Exception:
            traceback.print_exc()

    def _build_DVL_plot(self):
        self._DVL_event_sub = None
        self._DVL_x_vel = []
        self._DVL_y_vel = []
        self._DVL_z_vel = []

        kwargs = {
            "label": "DVL reading xyz vel (m/s)",
            "on_clicked_fn": self.toggle_DVL_step,
            "data": [self._DVL_x_vel, self._DVL_y_vel, self._DVL_z_vel],
        }
        (
            self._DVL_plot,
            self._DVL_plot_value,
        ) = combo_cb_xyz_plot_builder(**kwargs)
    def toggle_DVL_step(self, val=None):
        print("DVL DAQ: ", val)
        if val:
            if not self._DVL_event_sub:
                self._DVL_event_sub = (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_DVL_step)
                )
            else:
                self._DVL_event_sub = None
        else:
            self._DVL_event_sub = None



    def _on_DVL_step(self, e: carb.events.IEvent):
        # Casting np.float32 to float32 is necessary for the ui.Plot expects a consistent data type flow
        x_vel = float(self._scenario._DVL_reading[0])
        y_vel = float(self._scenario._DVL_reading[1])
        z_vel = float(self._scenario._DVL_reading[2])

        self._DVL_plot_value[0].set_value(x_vel)
        self._DVL_plot_value[1].set_value(y_vel)
        self._DVL_plot_value[2].set_value(z_vel)

        self._DVL_x_vel.append(x_vel)
        self._DVL_y_vel.append(y_vel)
        self._DVL_z_vel.append(z_vel)
        if len(self._DVL_x_vel) > 50:
            self._DVL_x_vel.pop(0)
            self._DVL_y_vel.pop(0)
            self._DVL_z_vel.pop(0)

        self._DVL_plot[0].set_data(*self._DVL_x_vel)
        self._DVL_plot[1].set_data(*self._DVL_y_vel)
        self._DVL_plot[2].set_data(*self._DVL_z_vel)

    def _build_baro_plot(self):
        self._baro_event_sub = None
        self._baro_data = []

        kwargs = {
                "label": "Barometer reading (Pa)", 
                "on_clicked_fn": self.toggle_baro_step, 
                "data": self._baro_data,
                "min": 101325.0,
                'max': 101325.0 + 50000,
                  }
        self._baro_plot, self._baro_plot_value = combo_cb_plot_builder(**kwargs)


    def toggle_baro_step(self, val=None):
        print('Barometer DAQ: ', val)
        if val:
            if not self._baro_event_sub:
                self._baro_event_sub= (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_baro_step)
                )
            else:
                self._baro_event_sub = None
        else:
            self._baro_event_sub = None

    def _on_baro_step(self, e: carb.events.IEvent):
        baro = float(self._scenario._baro_reading)
        self._baro_plot_value.set_value(baro)
        self._baro_data.append(baro)
        if len(self._baro_data) > 50:
            self._baro_data.pop(0)
        self._baro_plot.set_data(*self._baro_data)

    def _build_gyro_plot(self):
        self._IMU_event_sub_gyro = None
        self._gyro_x = []
        self._gyro_y = []
        self._gyro_z = []

        kwargs = {
            "label": "Gyroscope reading (rad/s)",
            "on_clicked_fn": self.toggle_gyro_step,
            "data": [self._gyro_x, self._gyro_y, self._gyro_z],
        }
        (
            self._gyro_plot,
            self._gyro_plot_value,
        ) = combo_cb_xyz_plot_builder(**kwargs)

    def toggle_gyro_step(self, val=None):
        print("Gyroscope DAQ: ", val)
        if val: # Checkbox is checked ON
            if not self._IMU_event_sub_gyro:
                self._IMU_event_sub_gyro = (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_gyro_step)
                )
        else: # Checkbox is checked OFF
            self._IMU_event_sub_gyro = None

    def _on_gyro_step(self, e: carb.events.IEvent):
        # Casting np.float32 to float32 is necessary for the ui.Plot expects a consistent data type flow
        x_gyro = float(self._scenario._IMU_reading['angular_velocity'][0])
        y_gyro = float(self._scenario._IMU_reading['angular_velocity'][1])
        z_gyro = float(self._scenario._IMU_reading['angular_velocity'][2])

        self._gyro_plot_value[0].set_value(x_gyro)
        self._gyro_plot_value[1].set_value(y_gyro)
        self._gyro_plot_value[2].set_value(z_gyro)

        self._gyro_x.append(x_gyro)
        self._gyro_y.append(y_gyro)
        self._gyro_z.append(z_gyro)
        if len(self._gyro_x) > 50:
            self._gyro_x.pop(0)
            self._gyro_y.pop(0)
            self._gyro_z.pop(0)

        self._gyro_plot[0].set_data(*self._gyro_x)
        self._gyro_plot[1].set_data(*self._gyro_y)
        self._gyro_plot[2].set_data(*self._gyro_z)

    def _build_accel_plot(self):
        self._IMU_event_sub_accel = None
        self._accel_x = []
        self._accel_y = []
        self._accel_z = []

        kwargs = {
            "label": "Accelerometer reading (m/s²)",
            "on_clicked_fn": self.toggle_accel_step,
            "data": [self._accel_x, self._accel_y, self._accel_z],
        }
        (
            self._accel_plot,
            self._accel_plot_value,
        ) = combo_cb_xyz_plot_builder(**kwargs)

    def toggle_accel_step(self, val=None):
        print("Accelerometer DAQ: ", val)
        if val:
            if not self._IMU_event_sub_accel:
                self._IMU_event_sub_accel = (
                    omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_accel_step)
                )
            else:
                self._IMU_event_sub_accel = None
        else:
            self._IMU_event_sub_accel = None

    def _on_accel_step(self, e: carb.events.IEvent):
        # Casting np.float32 to float32 is necessary for the ui.Plot expects a consistent data type flow
        x_accel = float(self._scenario._IMU_reading['linear_acceleration'][0])
        y_accel = float(self._scenario._IMU_reading['linear_acceleration'][1])
        z_accel = float(self._scenario._IMU_reading['linear_acceleration'][2])

        self._accel_plot_value[0].set_value(x_accel)
        self._accel_plot_value[1].set_value(y_accel)
        self._accel_plot_value[2].set_value(z_accel)

        self._accel_x.append(x_accel)
        self._accel_y.append(y_accel)
        self._accel_z.append(z_accel)
        if len(self._accel_x) > 50:
            self._accel_x.pop(0)
            self._accel_y.pop(0)
            self._accel_z.pop(0)

        self._accel_plot[0].set_data(*self._accel_x)
        self._accel_plot[1].set_data(*self._accel_y)
        self._accel_plot[2].set_data(*self._accel_z)


        
