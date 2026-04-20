import sys
from pathlib import Path

import numpy as np
from numpy import interp
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.random import uniform 
import random
from math import cos, sin, atan2, sqrt, pi
from scipy.spatial.transform import Rotation
import mujoco

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gym_rotor.envs.quad_utils import *
from typing import Optional
import args_parse

_MUJOCO_SIM_DIR = Path(__file__).resolve().parent / "mujoco-wheeled-uav-simulator"
if str(_MUJOCO_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_MUJOCO_SIM_DIR))

from qav_wheel.config import load_vehicle_params
from qav_wheel.model_builder import build_rotor_specs, build_uav_model_specs, render_model_xml
from qav_wheel.paths import DEFAULT_PATH_RESOLVER
from qav_wheel.simulation import build_sensor_layouts

class QuadEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None): 
        # Hyperparameters:
        parser = args_parse.create_parser()
        args = parser.parse_args()
        self.render_mode = render_mode if render_mode is not None else ("human" if args.render else None)

        self.vehicle_params = load_vehicle_params()
        self.rotor_specs = build_rotor_specs(self.vehicle_params)

        drone_params = self.vehicle_params["drone"]
        simulation_params = self.vehicle_params["simulation"]
        actuation_params = self.vehicle_params["actuation"]

        # Nominal value of quadrotor parameters:
        self.m_nominal = float(drone_params.get("mass", drone_params["body_box"]["mass"] + 2.0 * drone_params["wheels"]["mass"]))
        self.d_nominal = float(max(abs(float(drone_params["arm"]["x"])), abs(float(drone_params["arm"]["y"]))))
        inertia_matrix = np.asarray(drone_params["inertia"], dtype=float)
        self.J_nominal = inertia_matrix if inertia_matrix.shape == (3, 3) else np.diag(inertia_matrix.reshape(3))
        self.c_tf_nominal = float(actuation_params.get("yaw_moment_ratio", 0.0135))
        self.c_tw_nominal = float(actuation_params.get("max_rotor_thrust", 20.0))
        self.g = abs(float(simulation_params["gravity"][2]))

        self.m = self.m_nominal
        self.d = self.d_nominal
        self.J = np.array(self.J_nominal, dtype=float)
        self.c_tf = self.c_tf_nominal
        self.c_tw = self.c_tw_nominal

        # Force and Moment:
        self.f = self.m_nominal * self.g # magnitude of total thrust to overcome  
                                 # gravity and mass (No air resistance), [N]
        self.hover_force = self.m_nominal * self.g/ 4.0 # thrust magnitude of each motor, [N]
        self.min_force = 0.0 # minimum thrust of each motor, [N]
        self.max_force = float(actuation_params.get("max_rotor_thrust", self.c_tw_nominal * self.hover_force)) # maximum thrust of each motor, [N]
        self.avrg_act = (self.min_force+self.max_force)/2.0 
        self.scale_act = self.max_force-self.avrg_act # actor scaling

        self.f1 = self.hover_force # thrust of each 1st motor, [N]
        self.f2 = self.hover_force # thrust of each 2nd motor, [N]
        self.f3 = self.hover_force # thrust of each 3rd motor, [N]
        self.f4 = self.hover_force # thrust of each 4th motor, [N]
        self.M  = np.zeros(3) # magnitude of moment on quadrotor, [Nm]

        self.fM = np.zeros((4, 1)) # Force-moment vector
        self.forces_to_fM = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [self.rotor_specs[0].position[1], self.rotor_specs[1].position[1], self.rotor_specs[2].position[1], self.rotor_specs[3].position[1]],
            [-self.rotor_specs[0].position[0], -self.rotor_specs[1].position[0], -self.rotor_specs[2].position[0], -self.rotor_specs[3].position[0]],
            [self.rotor_specs[0].spin_sign * self.rotor_specs[0].yaw_moment_ratio, self.rotor_specs[1].spin_sign * self.rotor_specs[1].yaw_moment_ratio, self.rotor_specs[2].spin_sign * self.rotor_specs[2].yaw_moment_ratio, self.rotor_specs[3].spin_sign * self.rotor_specs[3].yaw_moment_ratio]
        ]) # Conversion matrix of forces to force-moment 
        self.fM_to_forces = np.linalg.inv(self.forces_to_fM)

        # Simulation parameters:
        self.freq = 200 # frequency [Hz]
        self.dt = 1./self.freq # discrete timestep, t(2) - t(1), [sec]
        self.ode_integrator = "mujoco" # physics backend
        self.R2D = 180./pi # [rad] to [deg]
        self.D2R = pi/180. # [deg] to [rad]
        self.e1 = np.array([1.,0.,0.])
        self.e2 = np.array([0.,1.,0.])
        self.e3 = np.array([0.,0.,1.])
        self.use_UDM = args.use_UDM # uniform domain randomization for sim-to-real transfer
        self.UDM_percentage = args.UDM_percentage

        # Coefficients in reward function:
        self.framework = args.framework
        self.reward_alive = 0.  # ≥ 0 is a bonus value earned by the agent for staying alive
        self.reward_crash = -1. # Out of boundary or crashed!
        self.Cx = args.Cx
        self.CIx = args.CIx
        self.Cv = args.Cv
        self.Cb1 = args.Cb1
        self.CIb1 = args.CIb1
        self.CW = args.Cw12
        self.reward_min = -np.ceil(self.Cx+self.CIx+self.Cv+self.Cb1+self.CIb1+self.CW)
        if self.framework in ("CMP","DMP"):
            # Agent1's reward:
            self.Cw12 = args.Cw12
            self.reward_min_1 = -np.ceil(self.Cx+self.CIx+self.Cv+self.Cw12)
            # Agent2's reward:
            self.CW3 = args.CW3
            self.reward_min_2 = -np.ceil(self.Cb1+self.CW3+self.CIb1)
        
        # Integral terms:
        self.sat_sigma = 1.
        self.eIx = IntegralErrorVec3() # Position integral error
        self.eIb1 = IntegralError() # Yawing integral error
        self.eIx.set_zero() # Set all integrals to zero
        self.eIb1.set_zero()

        # Commands:
        self.xd  = np.array([0.,0.,0.]) # desired tracking position command, [m] 
        self.vd  = np.array([0.,0.,0.]) # [m/s]
        self.b1d = np.array([1.,0.,0.]) # desired heading direction        
        self.Wd  = np.array([0.,0.,0.]) # desired angular velocity [rad/s]

        # Limits of states:
        self.x_lim = 1.0 # [m]
        self.v_lim = 4.0 # [m/s]
        self.W_lim = 2*pi # [rad/s]
        self.euler_lim = 85 # [deg]
        self.low = np.concatenate([-self.x_lim * np.ones(3),  
                                   -self.v_lim * np.ones(3),
                                   -np.ones(9),
                                   -self.W_lim * np.ones(3)],
                                   dtype=np.float32) #np.float64
        self.high = np.concatenate([self.x_lim * np.ones(3),  
                                    self.v_lim * np.ones(3),
                                    np.ones(9),
                                    self.W_lim * np.ones(3)],
                                    dtype=np.float32) #np.float64

        # Observation space:
        self.observation_space = spaces.Box(
            low=self.low, 
            high=self.high, 
            dtype=np.float32, #np.float64
        )

        # Action space:
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(4,),
            dtype=np.float32, #np.float64
        ) 

        self.model_xml_path, _, self.uav_specs = render_model_xml(
            self.vehicle_params,
            instance_id=0,
            path_resolver=DEFAULT_PATH_RESOLVER,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.model_xml_path))
        self.data = mujoco.MjData(self.model)
        self.sensor_layouts = build_sensor_layouts(self.model, build_uav_model_specs(self.vehicle_params, 1))
        self.sensor_layout = self.sensor_layouts[0]
        self._physics_steps_per_env_step = max(1, int(round(self.dt / float(self.model.opt.timestep))))
        self._body_name = str(drone_params["name"])

        # Init:
        self.state = None
        self.viewer = None
        self.screen_width = 1080
        self.screen_height = 480
        self.screen_capture = 1 

        self.ctrl = np.zeros(4, dtype=float)

    def _sync_control_state(self):
        self.ctrl = np.asarray(self.ctrl, dtype=float).reshape(4)
        self.f1, self.f2, self.f3, self.f4 = self.ctrl
        self.fM = self.forces_to_fM @ self.ctrl
        self.f = float(self.fM[0])
        self.M = np.asarray(self.fM[1:4], dtype=float)

    def _rotation_matrix_to_quat_wxyz(self, rotation_matrix: np.ndarray) -> np.ndarray:
        quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
        return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)

    def _quat_wxyz_to_rotation_matrix(self, quat_wxyz: np.ndarray) -> np.ndarray:
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=float)
        return Rotation.from_quat(quat_xyzw).as_matrix()

    def _sync_state_from_mujoco(self):
        position = np.array(self.data.sensordata[self.sensor_layout.position], dtype=float)
        linear_velocity = np.array(self.data.sensordata[self.sensor_layout.linear_velocity], dtype=float)
        angular_velocity_world = np.array(self.data.sensordata[self.sensor_layout.angular_velocity], dtype=float)
        rotation_matrix = np.column_stack(
            (
                self.data.sensordata[self.sensor_layout.x_axis],
                self.data.sensordata[self.sensor_layout.y_axis],
                self.data.sensordata[self.sensor_layout.z_axis],
            )
        )
        angular_velocity_body = rotation_matrix.T @ angular_velocity_world
        self.state = np.concatenate((position, linear_velocity, rotation_matrix.reshape(9, order='F'), angular_velocity_body), axis=0)
        return self.state

    def _advance_physics(self):
        self.data.ctrl[:] = self.ctrl
        for _ in range(self._physics_steps_per_env_step):
            mujoco.mj_step(self.model, self.data)
        self._sync_state_from_mujoco()


    def step(self, normalized_action):
        # Action:
        action = self.action_wrapper(normalized_action)
        self._sync_control_state()
        self._advance_physics()

        # Observation:
        obs = self.observation_wrapper(self.state)

        # Reward function:
        reward = self.reward_wrapper(obs)
        if self.framework in ("CMP","DMP"):
            if not isinstance(reward, (list, tuple, np.ndarray)):
                reward = [float(reward), float(reward)]
            else:
                reward = list(reward)
            reward[0] = interp(reward[0], [self.reward_min_1, 0.], [0., 1.]) # linear interpolation [0,1]
            reward[1] = interp(reward[1], [self.reward_min_2, 0.], [0., 1.]) # linear interpolation [0,1]
        elif self.framework == "NMP":
            reward_value = float(reward[0] if isinstance(reward, (list, tuple, np.ndarray)) else reward)
            reward = float(interp(reward_value, [self.reward_min, 0.], [0., 1.])) # linear interpolation [0,1]

        # Terminal condition:
        done = self.done_wrapper(obs)
        if isinstance(done, (list, tuple, np.ndarray)):
            if done[0]: # Out of boundary or crashed!
                if isinstance(reward, list):
                    reward[0] = self.reward_crash
            if self.framework in ("CMP","DMP") and done[1]: # Out of boundary or crashed!
                if isinstance(reward, list):
                    reward[1] = self.reward_crash
        elif done:
            reward = self.reward_crash

        return obs, reward, done, False, {}


    def reset(self, 
        env_type='train',
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        # Domain randomization:
        self.set_random_parameters(env_type) if self.use_UDM else None

        mujoco.mj_resetData(self.model, self.data)

        # Initial state error:
        self.sample_init_error(env_type)

        base_position = np.array(self.vehicle_params["drone"].get("initial_position", [0.0, 0.0, 0.0]), dtype=float)

        # x, position:
        position = base_position + uniform(size=3,low=-self.init_x,high=self.init_x)

        # v, velocity:
        linear_velocity = uniform(size=3,low=-self.init_v,high=self.init_v)

        # W, angular velocity:
        angular_velocity_body = uniform(size=3,low=-self.init_W,high=self.init_W)

        # R, attitude:
        roll_pitch = uniform(size=2,low=-self.init_R,high=self.init_R)
        euler = np.concatenate((roll_pitch, self.yaw), axis=None)
        R = Rotation.from_euler('xyz', euler, degrees=False).as_matrix()
        # Re-orthonormalize:
        if not isRotationMatrix(R):
            U, s, VT = psvd(R)
            R = U @ VT.T
        quat_wxyz = self._rotation_matrix_to_quat_wxyz(R)

        self.data.qpos[0:3] = position
        self.data.qpos[3:7] = quat_wxyz
        self.data.qvel[0:3] = linear_velocity
        self.data.qvel[3:6] = R @ angular_velocity_body
        mujoco.mj_forward(self.model, self.data)
        self._sync_state_from_mujoco()
        
        # Reset forces & moments:
        self.ctrl = np.full(4, self.hover_force, dtype=float)
        self._sync_control_state()

        # Integral terms:
        self.eIx.set_zero() # Set all integrals to zero
        self.eIb1.set_zero()

        # for drawing real-time plots:
        self.t = 0
        self.cmd_count = 0

        return np.array(self.state, dtype=np.float32)


    def action_wrapper(self, normalized_action):
        # Linear scale, normalized_action in [-1, 1] -> [min_act, max_act] 
        action = np.asarray(
            self.scale_act * normalized_action + self.avrg_act
            , dtype=float).clip(self.min_force, self.max_force)

        # Saturated thrust of each motor:
        self.ctrl = action

        return action


    def observation_wrapper(self, state):
        return np.array(self.state, dtype=np.float32)
    

    def reward_wrapper(self, obs):
        # Decomposing state vectors
        x, v, R, W = state_decomposition(obs)

        # Reward function coefficients:
        Cx = self.Cx # pos coef.
        Cv = self.Cv # vel coef.
        Cb1 = self.Cb1 # heading coef.
        CW = self.CW # ang_vel coef.

        # Errors:
        eX = x - self.xd # position error
        eV = v - self.vd # velocity error
        eb1 = norm_ang_btw_two_vectors(self.b1d, get_current_b1(R)) # heading errors

        # Reward function:
        reward_eX = -Cx*(norm(eX, 2)**2) 
        reward_eV = -Cv*(norm(eV, 2)**2)
        reward_eb1 = -Cb1*(abs(eb1))
        reward_eW = -CW*(norm(W, 2)**2)

        reward = self.reward_alive + (reward_eX + reward_eb1 + reward_eV + reward_eW)
        #reward *= 0.1 # rescaled by a factor of 0.1

        return reward


    def done_wrapper(self, obs):
        # Decomposing state vectors
        x, v, R, W = state_decomposition(obs)

        # Convert rotation matrix to Euler angles:
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        #eulerAngles = rotationMatrixToEulerAngles(R) * self.R2D

        done = False
        done = bool(
               (abs(x) >= self.x_lim).any() # [m]
            or (abs(v) >= self.v_lim).any() # [m/s]
            or (abs(W) >= self.W_lim).any() # [rad/s]
            or abs(euler[0]) >= self.euler_lim # phi
            or abs(euler[1]) >= self.euler_lim # theta
        )

        return done


    def EoM(self, t, state):
        # Decomposing state vectors
        x, v, R, W = state_decomposition(state)

        # Equations of motion of the quadrotor UAV
        x_dot = v
        v_dot = self.g*self.e3 - self.f*R@self.e3/self.m
        R_vec_dot = (R@hat(W)).reshape(1, 9, order='F')
        W_dot = inv(self.J)@(-hat(W)@self.J@W + self.M)
        state_dot = np.concatenate([x_dot.flatten(), 
                                    v_dot.flatten(),                                                                          
                                    R_vec_dot.flatten(),
                                    W_dot.flatten()])

        return np.array(state_dot)


    def sample_init_error(self, env_type='train'):
        self.yaw = uniform(size=1,low=-pi, high=pi)  # initial yaw angle, [rad]
        if env_type == 'train':
            # Spawning at the origin position and at zero angle (w/ random linear and angular velocity).
            if random.random() < 0.2: # 20% of the training
                self.init_x = 0.0 # initial pos error,[m]
                self.init_v = 0. # initial vel error, [m/s]
                self.init_R = 0 * self.D2R  # ±0 deg 
                self.init_W = 0. # initial ang vel error, [rad/s]
            else:
                self.init_x = 0.6 # initial pos error,[m]
                self.init_v = self.v_lim*0.5 # 50%; initial vel error, [m/s]
                self.init_R = 50 * self.D2R  # ±50 deg 
                self.init_W = self.W_lim*0.5 # 50%; initial ang vel error, [rad/s]
        elif env_type == 'eval':
            self.init_x = 0.4 # initial pos error,[m]
            self.init_v = self.v_lim*0.0 # initial vel error, [m/s]
            self.init_R = 0 * self.D2R # ±5 deg
            self.init_W = self.W_lim*0.0 # initial ang vel error, [rad/s]


    def set_random_parameters(self, env_type='train'):
        # Keep the MuJoCo model parameters consistent with the shared simulator config.
        self.m = self.m_nominal
        self.d = self.d_nominal
        self.J = np.array(self.J_nominal, dtype=float)
        self.c_tf = self.c_tf_nominal
        self.c_tw = self.c_tw_nominal

        self.f = self.m * self.g
        self.hover_force = self.m * self.g / 4.0
        self.min_force = 0.0
        self.max_force = float(self.vehicle_params["actuation"].get("max_rotor_thrust", self.max_force))
        self.fM = np.zeros(4, dtype=float)
        self.avrg_act = (self.min_force + self.max_force) / 2.0
        self.scale_act = self.max_force - self.avrg_act

        # print('m:',f'{self.m:.3f}','d:',f'{self.d:.3f}','J:',f'{J1:.4f}',f'{J3:.4f}','c_tf:',f'{self.c_tf:.4f}','c_tw:',f'{self.c_tw:.3f}')
        

    def get_current_state(self):
        return self.state


    def set_goal_state(self, xd, vd, b1d, b1d_dot, Wd):
        self.xd  = xd # desired tracking position command, [m] 
        self.vd  = vd # desired velocity command, [m/s]
        self.b1d = b1d # desired heading direction   
        self.b1d_dot = b1d_dot # desired heading direction derivative     
        self.Wd  = Wd # desired angular velocity [rad/s]


    def get_norm_error_state(self, framework):
        # Normalize state vectors: [max, min] -> [-1, 1]
        x_norm, v_norm, R_vec, W_norm = state_normalization(self.state, self.x_lim, self.v_lim, self.W_lim)
        R = R_vec.reshape(3, 3, order='F')

        # Normalize goal state vectors: [max, min] -> [-1, 1]
        xd_norm = self.xd/self.x_lim
        vd_norm = self.vd/self.v_lim
        Wd_norm = self.Wd/self.W_lim

        # Normalized error obs:
        ex_norm = x_norm - xd_norm # norm pos error
        ev_norm = v_norm - vd_norm # norm vel error
        eW_norm = W_norm - Wd_norm # norm ang vel error
        eW3_norm = W_norm[2] - Wd_norm[2]

        # Compute yaw angle error: 
        b1, b2, b3 = R@self.e1, R@self.e2, R@self.e3
        '''
        b1c = -(hat(b3) @ hat(b3)) @ self.b1d # desired b1
        eb1_norm = norm_ang_btw_two_vectors(b1c, b1) # b1 error, [-1, 1)
        '''
        b1c = self.b1d - np.dot(self.b1d, b3) * b3
        eb1 = np.arctan2(-np.dot(b1c, b2), np.dot(b1c, b1))
        eb1_norm = eb1/np.pi
        
        # Update integral terms: 
        self.eIx.integrate(-self.alpha*self.eIx.error + ex_norm*self.x_lim, self.dt) 
        self.eIx_norm = clip(self.eIx.error/self.eIx_lim, -self.sat_sigma, self.sat_sigma)
        self.eIb1.integrate(-self.beta*self.eIb1.error + eb1_norm*np.pi, self.dt) # b1 integral error
        self.eIb1_norm = clip(self.eIb1.error/self.eIb1_lim, -self.sat_sigma, self.sat_sigma)

        if framework in ("CMP","DMP"):
            # Agent1's obs:
            ew12_norm = eW_norm[0]*b1 + eW_norm[1]*b2
            obs_1 = np.concatenate((ex_norm, self.eIx_norm, ev_norm, b3, ew12_norm), axis=None, dtype=np.float32)
            # Agent2's obs:
            '''
            eW3_norm = eW_norm[2]
            '''
            obs_2 = np.concatenate((b1, eb1_norm, self.eIb1_norm, eW3_norm), axis=None, dtype=np.float32)
            error_obs_n = [obs_1, obs_2]
        elif framework == "NMP":
            # Single-agent's obs:
            R_vec = R.reshape(9, 1, order='F').flatten()
            obs = np.concatenate((ex_norm, self.eIx_norm, ev_norm, R_vec, eb1_norm, self.eIb1_norm, eW_norm), axis=None, dtype=np.float32)
            error_obs_n = [obs]
        
        return error_obs_n


    def render(self, mode='human', close=False):
        if self.state is None:
            return None

        if self.render_mode != "human":
            return False

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.lookat[:] = np.asarray(self.state[0:3], dtype=float)
            viewer.cam.distance = 4.0
            viewer.cam.elevation = -20.0
            viewer.cam.azimuth = 45.0
            viewer.sync()
        return True


    def close(self):
        if self.viewer:
            self.viewer = None