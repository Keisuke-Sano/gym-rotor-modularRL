import numpy as np
from numpy.linalg import norm

from gym_rotor.envs.quad import QuadEnv
from gym_rotor.envs.quad_utils import *
from gym_rotor.wrappers.wrapper_utils import *
from typing import Optional
import args_parse

class CoupledWrapper(QuadEnv):
    # metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: Optional[str] = None): 
        super().__init__()

        # Hyperparameters:
        parser = args_parse.create_parser()
        args = parser.parse_args()
        self.alpha, self.beta = args.alpha, args.beta # addressing noise or delay

        # limits of states:
        self.eIx_lim  = 3.0 
        self.eIb1_lim = 3.0 


    def reset(self, 
        env_type='train',
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        # super().reset(seed=seed)
        QuadEnv.reset(self, env_type)

        # Reset integral terms:
        self.eIx.set_zero() # Set all integrals to zero
        self.eIx_norm = np.zeros(3)
        self.eIb1.set_zero()
        self.eIb1_norm = 0.
        
        return np.array(self.state, dtype=np.float32)


    def action_wrapper(self, action):
        # Linear scale, [-1, 1] -> [min_act, max_act] 
        f_total = (
            4 * (self.scale_act * action[0] + self.avrg_act)
            ).clip(4*self.min_force, 4*self.max_force)

        self.fM = np.array([f_total, action[1], action[2], action[3]], dtype=float)
        self.ctrl = self.fM_to_forces @ self.fM
        
        return action


    def observation_wrapper(self, state):
        # Monolithic agent's obs:
        """
        norm_obs = (ex_norm, eIx_norm, ev_norm, R_vec, eb1_norm, eIb1_norm, eW_norm)
        """
        obs = self.get_norm_error_state(self.framework)

        return obs
    

    def reward_wrapper(self, obs):
        # Single-agent's obs:
        ex_norm, eIx_norm, ev_norm, _, eb1_norm, eIb1_norm, eW_norm = obs_decomposition(obs[0]) 

        # Single-agent's reward:
        reward_eX   = -self.Cx*(norm(ex_norm, 2)**2) 
        reward_eIX  = -self.CIx*(norm(eIx_norm, 2)**2)
        reward_eV   = -self.Cv*(norm(ev_norm, 2)**2)
        reward_eb1  = -self.Cb1*abs(eb1_norm)
        reward_eIb1 = -self.CIb1*(abs(eIb1_norm)**2)
        reward_eW   = -self.CW*(norm(eW_norm, 2)**2)
        
        rwd = reward_eX + reward_eIX + reward_eV + reward_eb1 + reward_eIb1 + reward_eW
        
        return [rwd]


    def done_wrapper(self, obs):
        # Single-agent's obs:
        ex_norm, eIx_norm, ev_norm, _, eb1_norm, eIb1_norm, eW_norm = obs_decomposition(obs[0]) 

        # Single-agent's terminal states:
        done = False
        done = bool(
               (abs(ex_norm) >= 1.0).any() 
            or (abs(ev_norm) >= 1.0).any() 
            or (abs(eW_norm) >= 1.0).any() 
            # or (abs(eb1_norm) >= 1.0)
            # or (abs(eIx_norm) >= 1.0).any()
            # or (abs(eIb1_norm) >= 1.0)
        )

        return [done]