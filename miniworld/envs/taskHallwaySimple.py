import math

from gymnasium import spaces, utils
import numpy as np
from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv

class TaskHallwaySimple(MiniWorldEnv, utils.EzPickle):

    """
    ## Description 

    Environment in which the agent has to navigate through a very long hallway. 
    There is a reward at the end. 

     ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box reached

    ## Arguments

    min_section_length      : minimum length of the hallway. 

    max_section_length      : maximum length of the hallway. 
                              The length of the hallway is chosen randomly between min_section_length and max_section_length.

    max_episode_steps       : maximum number of steps in an episode.

    facing_forward          : if True, the agent will be placed such that it is facing the reward.

    reset_keep_same_length  : if True, the length of the hallway will be the same at each reset.

    wall_tex                : texture of the walls of the hallway.
    
    """

    def __init__(self,min_section_length=5,
                      max_section_length=10,
                      max_episode_steps=250,
                      facing_forward=True,
                      reset_keep_same_length=False,
                      wall_tex='stripes_big',
                        **kwargs):
        
        self.max_episode_steps = max_episode_steps
        self.min_section_length = min_section_length
        self.max_section_length = max_section_length
        self.facing_forward = facing_forward
        self.reset_keep_same_length = reset_keep_same_length
        self.wall_tex = wall_tex

        self.total_length = np.random.randint(self.min_section_length,self.max_section_length)
        print("hallway length : ", self.total_length)

        MiniWorldEnv.__init__(self, max_episode_steps=self.max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only movement actions (left/right/forward) => do we want to allow left / right actions?
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Create a long rectangular room

        if not self.reset_keep_same_length :
            self.total_length = np.random.randint(self.min_section_length,self.max_section_length)
            # print("hallway length : ", self.total_length)
        else :
            # print('keep same length')
            pass

        room = self.add_rect_room(min_x=-1, max_x=-1 + self.total_length,
                                  min_z=-1, max_z=1,
                                  wall_tex= self.wall_tex,
                                  floor_tex='asphalt',
                                  no_ceiling=True
                                  )
        
        # Place the box at the end of the hallway
        self.box = self.place_entity(Box(color="red"), min_x=room.max_x - 2)

        # if facing_forward, place the agent such that it is facing the reward
        if self.facing_forward :
            self.place_agent(dir=self.np_random.uniform(-0.01, 0.01), max_x= 1)
        else :
            self.place_agent(dir=self.np_random.uniform(-math.pi / 4, math.pi / 4), max_x= 1)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info

