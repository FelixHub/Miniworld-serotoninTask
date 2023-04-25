import math

from gymnasium import spaces, utils
import numpy as np
from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv

class TaskHallwayControl(MiniWorldEnv, utils.EzPickle):

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

    TaskHallwayControl(nb_sections=3,motor_gains=[1,1,1],sections_length=[5,5,10])

    nb_sections             : number of sections in the hallway. For each section, there is a probability that the motor gain will be different from 1.

    motor_gains             : list of the motor gains for each section

    sections_length : lengths of the hallway sections

    """

    def __init__(self,  nb_sections=3,
                        random_gain = False,
                        motor_gains=[1,0.5,2],
                        sections_length=[5,5,10],
                        max_episode_steps=100,
                        **kwargs):
        
        # if training, we want the agent to spawn randomly in the hallway, and not in the opposite side to the reward
        self.max_episode_steps = max_episode_steps
        
        self.nb_sections = nb_sections

        if random_gain :
            self.motor_gains = np.random.choice(motor_gains,nb_sections)
        else :
            self.motor_gains = motor_gains

        self.sections_limit = [-1]
        self.sections_length = []

        for s in range(self.nb_sections):
            length = sections_length[s]
            self.sections_limit.append(self.sections_limit[-1] + length)
            self.sections_length.append(length)

        self.sections_motor_gain = motor_gains
        self.total_length = self.sections_limit[-1]

        MiniWorldEnv.__init__(self, max_episode_steps=self.max_episode_steps, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only movement actions (left/right/forward) => do we want to allow left / right actions?
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(min_x=-1, max_x=-1 + self.total_length,
                                  min_z=-1, max_z=1,
                                  wall_tex='stripes_big',
                                  floor_tex='asphalt',
                                  no_ceiling=True
                                  )
        
        # Place the box at the end of the hallway
        self.box = self.place_entity(Box(color="red"), min_x=room.max_x - 2)

        self.place_agent(dir= 0, max_x= 1, min_z=0, max_z=0)

    def step(self, action):

        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info

