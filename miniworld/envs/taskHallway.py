import math

from gymnasium import spaces, utils
import numpy as np
from miniworld.entity import Box
from miniworld.miniworld import MiniWorldEnv

class TaskHallway(MiniWorldEnv, utils.EzPickle):

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

    TaskHallway(nb_sections=10,proba_change_motor_gain=0.1,min_section_length=5,max_section_length=10,motor_gains=[0.3,0.6,2,3])

    nb_sections             : number of sections in the hallway. For each section, there is a probability that the motor gain will be different from 1.

    proba_change_motor_gain : probability that the motor gain will change for a section. 
                              If the motor gain changes, it will be chosen randomly from the list of possible motor gains.

    motor_gains             : list of the possible motor gains to choose from.

    min_section_length      : minimum length of a section. 

    max_section_length      : maximum length of a section. 
                              The length of a section is chosen randomly between min_section_length and max_section_length.
    
    """

    def __init__(self, nb_sections=5,
                        proba_change_motor_gain=0.5,
                        motor_gains=[0.3,0.6,2,3],
                        min_section_length=5,
                        max_section_length=10,
                        training=False,
                        **kwargs):
        # if training, we want the agent to spawn randomly in the hallway, and not in the opposite side to the reward
        self.training = training
        
        self.nb_sections = nb_sections
        self.proba_change_motor_gain = proba_change_motor_gain
        self.motor_gains = motor_gains
        self.min_section_length = min_section_length
        self.max_section_length = max_section_length

        self.sections_limit = [-1]
        self.sections_length = []
        self.sections_motor_gain = []

        for s in range(self.nb_sections):
            length = np.random.randint(self.min_section_length,self.max_section_length)
            self.sections_limit.append(self.sections_limit[-1] + length)
            self.sections_length.append(length)
            if (s > 0) and (np.random.random() < self.proba_change_motor_gain):
                motor_gain = self.np_random.choice(self.motor_gains)
            else:
                motor_gain = 1
            self.sections_motor_gain.append(motor_gain)

        self.total_length = self.sections_limit[-1]

        print("sections limits", self.sections_limit)
        print("sections lengths", self.sections_length)
        print("sections motor gains", self.sections_motor_gain)

        MiniWorldEnv.__init__(self, max_episode_steps=250, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only movement actions (left/right/forward) => do we want to allow left / right actions ?
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

        # Place the agent a random distance away from the goal
        if self.training :
            self.place_agent(dir=self.np_random.uniform(-math.pi / 4, math.pi / 4), max_x= room.max_x - 2)
        else :
            self.place_agent(dir=self.np_random.uniform(-math.pi / 4, math.pi / 4), max_x= 1)

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info

