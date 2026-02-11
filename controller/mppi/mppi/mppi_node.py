import os
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".60"

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult, ParameterDescriptor, ParameterType, FloatingPointRange, IntegerRange
from f110_msgs.msg import LtplWpntArray

import yaml
from ament_index_python.packages import get_package_share_directory
import numpy as np
import tf_transformations
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import WpntArray
from visualization_msgs.msg import Marker, MarkerArray
import time

from config.config import (
    MPPIConfig,
    MPCConfig,
    dynamic_mppi_config,
)
from config.dynamics_config import load_dynamics_config_from_yaml
from .planner import DynamicMPPIPlanner

import jax

class MPPINode(Node):
    def __init__(self):
        super().__init__('mppi_node')

        self.get_logger().info("="*30)
        self.get_logger().info(f"JAX Backend: {jax.lib.xla_bridge.get_backend().platform}")
        self.get_logger().info(f"JAX Devices: {jax.devices()}")
        self.get_logger().info("="*30)
        
        # Declare ROS 2 Parameters
        self._declare_parameters()
        
        # Load MPPI Parameters from YAML explicitly
        self._load_mppi_params_from_yaml()
        
        # Load Dynamics Parameters
        self.dynamics_params = load_dynamics_config_from_yaml("models_param.yaml")
        
        # Create MPPI Config (Solver Settings)
        self.mppi_config = self._create_config_object()
        
        if self.dynamics_params is None or self.mppi_config is None:
            raise ValueError("Something went wrong while loading parameters.")
        
        # [수정] Model 객체 생성 (planner 초기화 전에!)
        from config.model import DynamicBicycleModel
        self.model = DynamicBicycleModel(self.dynamics_params)
        
        # Initialize Planner with explicit kwargs (이제 model이 준비됨)
        self.planner = DynamicMPPIPlanner(
            config=self.mppi_config, 
            params=self.dynamics_params,
            model=self.model  # ← model 추가
        )

        # --- JAX/Numba Warmup ---
        self.get_logger().info("Compiling JAX/Numba functions (Warmup) will happen after Map is received.")
        self.is_warmed_up = False
        # self._perform_warmup() # Moved to map_cb to ensure map exists
        # self.get_logger().info("Warmup Complete. Planner is ready.")

        self.add_on_set_parameters_callback(self.parameters_callback)

        # --- State Variables ---
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.yaw_rate = 0.0
        self.steering_angle = 0.0 
        self.beta = 0.0            
        self.local_waypoints = None
        
        # Track SDF Variables
        self.track_sdf = None
        self.map_data_jax = None
        self.map_metadata = None
        
        # Delay Compensator
        from .delay_compensator import DelayCompensator
        delay_time = self.mppi_config.delay_time  # From mppi_params.yaml
        wheelbase = self.dynamics_params.LF + self.dynamics_params.LR
        self.delay_compensator = DelayCompensator(delay_time, wheelbase, self.get_logger())
        self.get_logger().info(f"[DelayComp] Initialized with delay={delay_time:.3f}s, wheelbase={wheelbase:.3f}m")
        
        # --- Flags for startup checks ---
        self.pose_received = False
        self.odom_received = False
        self.wp_received = False


        # --- Publishers & Subscribers ---
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.vis_pub = self.create_publisher(MarkerArray, '/mppi/visualization', 1)
        self.costmap_pub = self.create_publisher(OccupancyGrid, '/mppi/cost_map', 1)
        
        self.pose_sub = self.create_subscription(PoseStamped, '/car_state/pose', self.pose_cb, 1)
        self.odom_sub = self.create_subscription(Odometry, '/car_state/odom', self.odom_cb, 1)
        self.wp_sub = self.create_subscription(WpntArray, '/local_waypoints', self.local_wp_cb, 1)
        
        # Map Subscriber
        self.map_sub = self.create_subscription(LtplWpntArray, '/ltpl_waypoints', self.map_cb, 10)

        # --- Wait for Messages ---
        self.wait_for_messages_()

        # --- Control Loop ---
        self.timer = self.create_timer(0.025, self.timer_cb)
        self.get_logger().info("MPPI Node Initialized and Running")

    def _declare_parameters(self):
        """
        Define and declare all ROS parameters for the MPPI node.
        """
        param_dicts = [
            # --- System ---
            {'name': 'dt', 'default': 0.025, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.01, to_value=0.2, step=0.001)])},
            {'name': 'predictive_horizon', 'default': 20, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, integer_range=[IntegerRange(from_value=10, to_value=80, step=1)])},
            
            # --- MPPI Hyperparameters ---
            {'name': 'n_samples', 'default': 2048, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, integer_range=[IntegerRange(from_value=512, to_value=8192, step=128)])},
            {'name': 'n_iterations', 'default': 1, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, integer_range=[IntegerRange(from_value=1, to_value=5, step=1)])},
            {'name': 'temperature', 'default': 0.1, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0001, to_value=100.0, step=0.0001)])},
            {'name': 'gamma', 'default': 0.01, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=0.2, step=0.001)])},
            
             # --- Weights (Clear & Intuitive) ---
            {'name': 'weight_contour', 'default': 2.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            {'name': 'weight_progress', 'default': 0.5, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            {'name': 'weight_slip', 'default': 0.001, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            
            {'name': 'weight_steer', 'default': 0.1, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            {'name': 'weight_accel', 'default': 0.01, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            
            {'name': 'weight_steer_rate', 'default': 10.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            {'name': 'weight_accel_rate', 'default': 1.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            
            {'name': 'terminal_weight_mult', 'default': 10.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            
            {'name': 'sampling_std', 'default': [1.0, 3.5], 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE_ARRAY)},
            
            # --- Constraints & Switches ---
            {'name': 'max_steer_angle', 'default': 0.4189, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            {'name': 'max_speed', 'default': 20.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            {'name': 'min_speed', 'default': 0.5, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE)},
            {'name': 'adaptive_covariance', 'default': True, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)},
            {'name': 'scan', 'default': True, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_BOOL)},
            {'name': 'delay_time', 'default': 0.05, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=1.0, step=0.01)])},
            {'name': 'collision_safety_margin', 'default': 0.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.0, to_value=2.0, step=0.01)])},
            {'name': 'costmap_contour_half_width', 'default': 0.5, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.05, to_value=2.0, step=0.01)])},
            {'name': 'costmap_contour_exp', 'default': 4.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=1.0, to_value=8.0, step=0.1)])},
            {'name': 'costmap_contour_scale', 'default': 1.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.1, to_value=100.0, step=0.1)])},
            {'name': 'costmap_wall_k', 'default': 5.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.1, to_value=20.0, step=0.1)])},
            {'name': 'costmap_wall_scale', 'default': 20.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=0.1, to_value=200.0, step=0.1)])},
            {'name': 'costmap_collision_cost', 'default': 10000.0, 'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_DOUBLE, floating_point_range=[FloatingPointRange(from_value=100.0, to_value=100000.0, step=100.0)])},
            
            # --- Debug / Visualization ---
            {
                'name': 'enable_visualization',
                'default': True,
                'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_BOOL),
            },
            {
                'name': 'profile_plan_breakdown',
                'default': False,
                'descriptor': ParameterDescriptor(type=ParameterType.PARAMETER_BOOL),
            },
        ]

        for param in param_dicts:
            self.declare_parameter(
                name=param['name'],
                value=param['default'],
                descriptor=param['descriptor']
            )

    def _load_mppi_params_from_yaml(self):
        """
        Load MPPI parameters from mppi_params.yaml if it exists and update ROS parameters.
        """
        file_name = "mppi_params.yaml"
        yaml_file = file_name
        
        try:
            share_dir = get_package_share_directory('mppi')
            candidate = os.path.join(share_dir, 'config', file_name)
            if os.path.exists(candidate):
                yaml_file = candidate
        except Exception:
            pass
        
        self.get_logger().info(f"Attempting to load config from: {yaml_file}")
            
        if not os.path.exists(yaml_file):
             current_dir = os.path.dirname(os.path.abspath(__file__))
             candidate = os.path.join(current_dir, '..', 'config', file_name)
             if os.path.exists(candidate):
                 yaml_file = candidate

        if os.path.exists(yaml_file):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                if 'mppi_node' in data and 'ros__parameters' in data['mppi_node']:
                    params_dict = data['mppi_node']['ros__parameters']
                    # Only update parameters that are already declared
                    ros_params = [Parameter(k, value=v) for k, v in params_dict.items() if self.has_parameter(k)]
                    if ros_params:
                        self.set_parameters(ros_params)
                    if ros_params:
                        self.set_parameters(ros_params)
                        self.get_logger().info(f"Loaded MPPI parameters from {yaml_file}")
            except Exception as e:
                self.get_logger().warn(f"Failed to load {yaml_file}: {e}")

    def _create_config_object(self):
        """
        ROS 파라미터를 읽어 MPPIConfig 인스턴스를 생성
        """
        # Array 값 읽기용 헬퍼
        def get_arr(name): return np.array(self.get_parameter(name).value)
        def get_val(name): return self.get_parameter(name).value

        # Retrieve Clear Weights
        w_contour = get_val('weight_contour')
        w_progress = get_val('weight_progress')
        w_slip = get_val('weight_slip')
        
        w_steer = get_val('weight_steer')
        w_accel = get_val('weight_accel')
        
        w_steer_rate = get_val('weight_steer_rate')
        w_accel_rate = get_val('weight_accel_rate')
        
        term_mult = get_val('terminal_weight_mult')

        # Construct Matrices for SB-MPCC
        # We repurpose the Q matrix diagonal for passing weights to cost.py
        # Q[0] -> Progress Weight (w_p)
        # Q[1] -> Contouring Weight (w_c)
        # Q[6] -> Slip Weight
        q_diag = np.zeros(7)
        q_diag[0] = w_progress
        q_diag[1] = w_contour
        q_diag[6] = w_slip
        
        Q_mat = np.diag(q_diag)
        
        # P: Terminal Cost
        P_mat = Q_mat * term_mult
        
        # R: Input Cost
        R_mat = np.diag([w_steer, w_accel])
        
        # Rd: Input Rate Cost
        Rd_mat = np.diag([w_steer_rate, w_accel_rate])

        # Map dynamics limits to MPC constraints
        # Control is now [delta, v] (absolute targets), not [delta_rate, accel]
        u_min = np.array([self.dynamics_params.MIN_STEER, self.dynamics_params.MIN_SPEED])
        u_max = np.array([self.dynamics_params.MAX_STEER, self.dynamics_params.MAX_SPEED])

        return MPPIConfig(
            # System
            dt=get_val('dt'),
            N=get_val('predictive_horizon'),
            nx=7, 
            nu=2,
            
            # Weights 
            Q=Q_mat,
            P=P_mat,
            R=R_mat,
            Rd=Rd_mat,
            
            # Constraints from Dynamics Params
            u_min=u_min,
            u_max=u_max,

            # Params
            n_samples=get_val('n_samples'),
            n_iterations=get_val('n_iterations'),
            temperature=get_val('temperature'),
            damping=get_val('gamma'),
            u_std=get_arr('sampling_std'),
            adaptive_covariance=get_val('adaptive_covariance'),
            scan=get_val('scan'),
            delay_time=get_val('delay_time'),
        )

    def _perform_warmup(self):
        """
        Run the planner once with dummy data to trigger JIT compilation.
        """
        # Dummy state: [x, y, delta, v, yaw, yaw_rate, beta]
        dummy_state = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        
        # Dummy waypoints: [x, y, psi, kappa, v, s]
        # Create a straight line to avoid division by zero in interpolation
        N_wp = 100
        dummy_waypoints = np.zeros((N_wp, 6))
        dummy_waypoints[:, 0] = np.linspace(0, 10, N_wp) # x increases from 0 to 10
        
        try:
            # Call plan to trigger JIT compilation for both Numba (interpolation) and JAX (solver)
            self.planner.plan(dummy_state, dummy_waypoints)
        except Exception as e:
            self.get_logger().warn(f"Warmup failed: {e}")

    def parameters_callback(self, params):
        success = True
        need_reinit = False

        for param in params:
            name = param.name
            val = param.value
            
            try:
                # Hyperparameters
                if name == 'temperature':
                    self.mppi_config.temperature = float(val)
                    need_reinit = True
                elif name == 'gamma':
                    self.mppi_config.damping = float(val)
                    need_reinit = True
                # Weights
                elif name in ['weight_contour', 'weight_progress', 'weight_slip', 'terminal_weight_mult']:
                    need_reinit = True 
                    
                # Input Weights
                elif name in ['weight_steer', 'weight_accel', 'weight_steer_rate', 'weight_accel_rate']:
                    need_reinit = True
                elif name == 'dt':
                    self.mppi_config.dt = float(val)
                    need_reinit = True
                
                # Weights (these are handled by _create_config_object, so just trigger reinit)
                elif name == 'q_diag': # This parameter is not declared, but included for completeness if it were.
                    self.mppi_config.Q = np.diag(np.array(val))
                    need_reinit = True
                elif name == 'r_diag': # This parameter is not declared, but included for completeness if it were.
                    self.mppi_config.R = np.diag(np.array(val))
                    need_reinit = True
                elif name == 'p_diag': # This parameter is not declared, but included for completeness if it were.
                    self.mppi_config.P = np.diag(np.array(val))
                    need_reinit = True
                elif name == 'rd_diag':
                    self.mppi_config.Rd = np.diag(np.array(val))
                    need_reinit = True
                elif name == 'sampling_std':
                    self.mppi_config.u_std = np.array(val)
                    need_reinit = True

                # Horizon & Sample
                elif name == 'predictive_horizon':
                    self.mppi_config.N = int(val)
                    need_reinit = True
                elif name == 'n_samples':
                    self.mppi_config.n_samples = int(val)
                    need_reinit = True
                elif name == 'delay_time':
                    self.mppi_config.delay_time = float(val)
                    need_reinit = True
                
                self.get_logger().info(f"Updated {name}: {val}")
                
            except Exception as e:
                self.get_logger().warn(f"Failed to update {name}: {e}")
                success = False

        if success:
            if need_reinit:
                self.get_logger().info("Re-initializing MPPI Planner with new parameters...")
                # 객체 재생성
                self.mppi_config = self._create_config_object() # Recreate config to pick up new weights
                self.planner = DynamicMPPIPlanner(config=self.mppi_config, params=self.dynamics_params)
                # Pass Map Data to Planner/Solver if available
                if self.planner.solver and self.map_data_jax:
                    self.planner.solver.set_map(self.map_data_jax, self.map_metadata)
            
            return SetParametersResult(successful=True)
        else:
            return SetParametersResult(successful=False, reason="Update Failed")

    # -------------------- Callbacks --------------------
    def pose_cb(self, msg):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        
        # Quaternion to Euler
        q = msg.pose.orientation
        q_list = [q.x, q.y, q.z, q.w]
        (roll, pitch, yaw) = tf_transformations.euler_from_quaternion(q_list)
        self.yaw = yaw
        self.pose_received = True

    def map_cb(self, msg):
        """
        Build TrackSDF when map data is received.
        """
        if self.track_sdf is not None:
            return # Only build once for now (or reset if needed)
            
        self.get_logger().info(f"Received Map Data: {len(msg.ltplwpnts)} waypoints. Baking SDF...")
        
        try:
            # Extract data
            wx = np.array([wp.x_ref_m for wp in msg.ltplwpnts])
            wy = np.array([wp.y_ref_m for wp in msg.ltplwpnts])
            wyaw = np.array([wp.psi_racetraj_rad for wp in msg.ltplwpnts])
            w_left = np.array([wp.width_left_m for wp in msg.ltplwpnts])
            w_right = np.array([wp.width_right_m for wp in msg.ltplwpnts])
            
            from .track_sdf import TrackSDF
            safety_margin = float(self.get_parameter('collision_safety_margin').value)
            contour_half_width = float(self.get_parameter('costmap_contour_half_width').value)
            contour_exp = float(self.get_parameter('costmap_contour_exp').value)
            contour_scale = float(self.get_parameter('costmap_contour_scale').value)
            wall_k = float(self.get_parameter('costmap_wall_k').value)
            wall_scale = float(self.get_parameter('costmap_wall_scale').value)
            collision_cost = float(self.get_parameter('costmap_collision_cost').value)
            self.track_sdf = TrackSDF(
                wx,
                wy,
                wyaw,
                w_left,
                w_right,
                resolution=0.05,
                map_padding=2.0,
                safety_margin=safety_margin,
                contour_half_width=contour_half_width,
                contour_exp=contour_exp,
                contour_scale=contour_scale,
                wall_k=wall_k,
                wall_scale=wall_scale,
                collision_cost=collision_cost,
            )
            
            # Visualize Map (Non-blocking)
            self.track_sdf.show_debug_map()
            self.get_logger().info("Map baked and displayed.")
            
            # Update Planner
            self.map_data_jax, self.map_metadata = self.track_sdf.get_jax_maps()
            if self.planner.solver:
                 self.planner.solver.set_map(self.map_data_jax, self.map_metadata)
            
            # Publish cost map for RViz visualization
            self.publish_costmap()
            
            # Trigger Warmup if not done yet
            if not self.is_warmed_up:
                self.get_logger().info("Map received. Triggering JAX/Numba Warmup...")
                try:
                    self._perform_warmup()
                    self.is_warmed_up = True
                    self.get_logger().info("Warmup Complete. Planner is ready.")
                except Exception as e:
                    self.get_logger().error(f"Warmup failed: {e}")
                 
        except Exception as e:
            self.get_logger().error(f"Failed to process map message: {e}")

    def odom_cb(self, msg):
        self.v = msg.twist.twist.linear.x
        self.yaw_rate = msg.twist.twist.angular.z
        self.odom_received = True

    def local_wp_cb(self, msg):
        # Check for empty waypoints
        if not msg.wpnts:
            return

        # Convert WpntArray to numpy array for planner
        # Pass raw waypoint data to planner: [x, y, psi, kappa, v, s]
        wps = []
        for wp in msg.wpnts:
            # x, y, psi(heading), kappa(curvature), v(velocity), s(progress)
            wps.append([wp.x_m, wp.y_m, wp.psi_rad, wp.kappa_radpm, wp.vx_mps, wp.s_m])
            
        self.local_waypoints = np.array(wps)
        self.wp_received = True

    def wait_for_messages_(self):
        """
        Wait for initial messages (pose, odom, waypoints) before starting the control loop.
        """
        self.get_logger().info("Waiting for required messages...")
        
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            
            if self.pose_received and self.odom_received and self.wp_received and self.is_warmed_up:
                break
            
        self.get_logger().info("All required messages received. Starting control loop.")

    def timer_cb(self):
        if self.local_waypoints is None:
            return

        dt = self.mppi_config.dt

        pred_x, pred_y, pred_yaw, pred_v = self.delay_compensator.compensate(
            self.x, self.y, self.yaw, self.v, dt
        )

        state = [pred_x, pred_y, self.steering_angle, pred_v, pred_yaw, self.yaw_rate, self.beta]

        enable_vis = bool(self.get_parameter('enable_visualization').value)
        do_profile = bool(self.get_parameter('profile_plan_breakdown').value)

        # Visualization at reduced rate (5 Hz) to avoid GPU sync every tick
        self._vis_counter = getattr(self, '_vis_counter', 0) + 1
        do_vis_this_tick = enable_vis and (self._vis_counter % 1 == 0)  # 40Hz / 8 = 5Hz

        t0 = time.time()
        target_speed_raw, target_steering_raw, opt_traj, sampled_trajs, ref_traj = self.planner.plan(
            state=state,
            waypoints=self.local_waypoints,
            params=self.dynamics_params,
            Q=self.mppi_config.Q,
            R=self.mppi_config.R,
            visualize=do_vis_this_tick,
        )
        t1 = time.time()

        # [Blocking 지점] 여기서만 GPU→CPU 동기화 발생
        t_convert_start = time.time()
        target_speed = float(target_speed_raw)
        target_steering = float(target_steering_raw)
        t2 = time.time()

        plan_time = (t1 - t0) * 1000.0
        t_convert = (t2 - t_convert_start) * 1000.0

        if do_profile:
            # If plan() returns DeviceArrays, nothing forces GPU sync until float().
            # This print helps verify whether the "gap" is actually JAX async work being waited on at conversion.
            self.get_logger().info(
                f"[ProfileBreakdown] plan()={plan_time:.2f}ms | convert(float)={t_convert:.2f}ms | total={(t2 - t0)*1000.0:.2f}ms",
                throttle_duration_sec=1.0,
            )

        # Clipping
        target_steering = np.clip(target_steering, self.dynamics_params.MIN_STEER, self.dynamics_params.MAX_STEER)
        target_speed = np.clip(target_speed, self.dynamics_params.MIN_SPEED, self.dynamics_params.MAX_SPEED)

        # Update delay compensator
        self.delay_compensator.update_queue(target_steering, target_speed, dt)
        self.steering_angle = target_steering

        # Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.header.frame_id = "base_link"
        drive_msg.drive.steering_angle = target_steering
        drive_msg.drive.speed = target_speed
        self.drive_pub.publish(drive_msg)

        # Publish Visualization (only on vis ticks — 5Hz)
        if do_vis_this_tick and opt_traj is not None:
            self.publish_visualization(opt_traj, sampled_trajs, ref_traj)

    def publish_visualization(self, opt_traj, sampled_trajs, ref_traj):
        marker_array = MarkerArray()
        timestamp = self.get_clock().now().to_msg()

        # 1. Optimal Trajectory (Green LineStrip)
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = timestamp
        marker.ns = "mppi_optimal"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.1 # Line width
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        # opt_traj shape: [N+1, 7] (x, y, delta, v, yaw, yaw_rate, beta)
        if opt_traj is not None:
            for i in range(opt_traj.shape[0]):
                p = Point()
                p.x = float(opt_traj[i, 0])
                p.y = float(opt_traj[i, 1])
                p.z = 0.1
                marker.points.append(p)
            marker_array.markers.append(marker)

        # 2. Sampled Trajectories (Blue LineLists, lighter)
        # sampled_trajs shape: [n_samples, N+1, 7]
        if sampled_trajs is not None:
            marker_s = Marker()
            marker_s.header.frame_id = "map"
            marker_s.header.stamp = timestamp
            marker_s.ns = "mppi_sampled"
            marker_s.id = 1
            marker_s.type = Marker.LINE_LIST
            marker_s.action = Marker.ADD
            marker_s.scale.x = 0.02
            marker_s.color.a = 0.2
            marker_s.color.r = 0.0
            marker_s.color.g = 0.5
            marker_s.color.b = 1.0

            # Visualize only a subset for performance (e.g., first 20 samples)
            num_vis_samples = min(20, sampled_trajs.shape[0])
            for i in range(num_vis_samples):
                traj = sampled_trajs[i]
                for j in range(traj.shape[0] - 1):
                    p1 = Point()
                    p1.x = float(traj[j, 0])
                    p1.y = float(traj[j, 1])
                    p1.z = 0.05

                    p2 = Point()
                    p2.x = float(traj[j + 1, 0])
                    p2.y = float(traj[j + 1, 1])
                    p2.z = 0.05

                    marker_s.points.append(p1)
                    marker_s.points.append(p2)
            marker_array.markers.append(marker_s)
        else:
            marker_clear_sampled = Marker()
            marker_clear_sampled.header.frame_id = "map"
            marker_clear_sampled.header.stamp = timestamp
            marker_clear_sampled.ns = "mppi_sampled"
            marker_clear_sampled.id = 1
            marker_clear_sampled.action = Marker.DELETE
            marker_array.markers.append(marker_clear_sampled)

        marker_clear_ref = Marker()
        marker_clear_ref.header.frame_id = "map"
        marker_clear_ref.header.stamp = timestamp
        marker_clear_ref.ns = "mppi_reference"
        marker_clear_ref.id = 2
        marker_clear_ref.action = Marker.DELETE
        marker_array.markers.append(marker_clear_ref)

        self.vis_pub.publish(marker_array)

    def publish_costmap(self):
        """
        Publish cost map as OccupancyGrid for RViz visualization.
        """
        if self.track_sdf is None:
            return
            
        try:
            msg = OccupancyGrid()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "map"
            
            # Map info
            msg.info.resolution = self.track_sdf.resolution
            msg.info.width = self.track_sdf.width_cells
            msg.info.height = self.track_sdf.height_cells
            msg.info.origin.position.x = self.track_sdf.min_x
            msg.info.origin.position.y = self.track_sdf.min_y
            msg.info.origin.position.z = 0.0
            
            # Convert binary cost map to occupancy values
            # cost_map: shape (W, H), transpose to (H, W) for row-major OccupancyGrid
            cost_map_np = np.array(self.track_sdf.cost_map.T)  # (H, W)
            
            # Binary: 0.0=free → 0, 1.0=wall → 100
            occupancy = (cost_map_np * 100).astype(np.int8)
            
            # Flatten to row-major (RViz expects this)
            msg.data = occupancy.flatten().tolist()
            
            self.costmap_pub.publish(msg)
            self.get_logger().info(f"[CostMap] Published {msg.info.width}x{msg.info.height} occupancy grid")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish cost map: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = MPPINode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()