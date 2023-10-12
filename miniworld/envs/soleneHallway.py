import math

from gymnasium import spaces, utils
import numpy as np
from miniworld.entity import Box, ImageFrame,TextFrame
from miniworld.miniworld import MiniWorldEnv


class SoleneHallway(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Environment in which the goal is to go to a red box at the end of a
    hallway within as few steps as possible.

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

    ```python
    Hallway(length=12)
    ```

    `length`: length of the entire space
    """

    def __init__(self, length=400,is_random=False, is_rewarded=True,is_ambiguous=False, **kwargs):
        assert length >= 2
        self.length = length
        self._size = length
        self.wall_tex = 'concrete' # not noise stripes_simple stripes_simple

        self.is_random = is_random
        print(kwargs)

        if is_random:
            self.is_rewarded = np.random.choice([True,False],p=[0.5,0.5])
            self.is_ambiguous = np.random.choice([True,False],p=[0.2,0.8])
        else :
            self.is_rewarded = is_rewarded
            self.is_ambiguous = is_ambiguous

        print("is_rewarded",is_rewarded,"is_ambiguous",is_ambiguous)


        
        MiniWorldEnv.__init__(self, max_episode_steps=50000, **kwargs)
        utils.EzPickle.__init__(self, length, **kwargs)

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        # Create a long rectangular room
        room = self.add_rect_room(min_x=-1, max_x= -1 + self.length + 100, min_z=-1, max_z=1,
                                wall_tex= self.wall_tex,
                                  floor_tex='asphalt',
                                  no_ceiling=True)

        # Place the agent a random distance away from the goal
        self.place_agent(
            dir=self.np_random.uniform(-0.0001, 0.0001), min_x=0.0001, max_x=0.0002,min_z=-0.0001, max_z=0.0001,
        )
        
        '''
        ### too complicated and the textures are too heterongenous +> we want simpler textures
        rewarded_textures = ["stripes_wide","triangle"]
        unrewarded_textures = ["stripes_wide_h","bubble"]
        ambiguous_texture = "floor_tiles_bw"
        reward_zone_texture = "white"
        '''

        unrewarded_textures = ["colorA","colorB"]
        rewarded_textures = ["colorC","colorD"]
        ambiguous_texture = "colorE"
        reward_zone_texture = "colorF"
        

        
        if self.is_ambiguous:
            first_zone_texture = ambiguous_texture
        elif self.is_rewarded:
             first_zone_texture = rewarded_textures[0]
        else:
            first_zone_texture = unrewarded_textures[0]
        if self.is_rewarded:
            second_zone_texture = rewarded_textures[1]
        else:
            second_zone_texture = unrewarded_textures[1]

        sign_1 = ImageFrame(
            pos=[0.2*self.length + (0.175*self.length/2), 1.5, 1],
            dir=math.pi / 2,
            tex_name=first_zone_texture,
            width=0.175*self.length,
            depth=0.01,
            height=3
        )
        sign_2 = ImageFrame(
            pos=[0.2*self.length + (0.175*self.length/2) , 1.5, -1],
            dir=- math.pi / 2,
            tex_name=first_zone_texture,
            width=0.175*self.length,
            depth=0.01,
            height=3
        )

        sign_3 = ImageFrame(
            pos=[0.55*self.length + (0.175*self.length/2), 1.5, 1],
            dir=math.pi / 2,
            tex_name=second_zone_texture,
            width=0.175*self.length,
            depth=0.01,
            height=3
        )
        sign_4 = ImageFrame(
            pos=[0.55*self.length + (0.175*self.length/2), 1.5, -1],
            dir=- math.pi / 2,
            tex_name=second_zone_texture,
            width=0.175*self.length,
            depth=0.01,
            height=3
        )

        sign_5 = ImageFrame(
            pos=[0.875*self.length + (0.05*self.length/2), 1.5, 1],
            dir=math.pi / 2,
            tex_name=reward_zone_texture,
            width=0.05*self.length,
            depth=0.01,
            height=3
        )
        sign_6 = ImageFrame(
            pos=[0.875*self.length + (0.05*self.length/2), 1.5, -1],
            dir=- math.pi / 2,
            tex_name=reward_zone_texture,
            width=0.05*self.length,
            depth=0.01,
            height=3
        )

        self.entities.append(sign_1)
        self.entities.append(sign_2)
        self.entities.append(sign_3)
        self.entities.append(sign_4)
        self.entities.append(sign_5)
        self.entities.append(sign_6)

    def step(self, action):
        old_pos_x = self.agent.pos[0]
        obs, reward, termination, truncation, info = super().step(action)
        new_pos_x = self.agent.pos[0]

        if old_pos_x <=  0.9*self.length < new_pos_x:
            if self.is_rewarded :
                print("rewarded!")
                reward += self._reward()
            else :
                print("unrewarded!")
                reward += 0
        if new_pos_x > self.length:
            termination = True

        return obs, reward, termination, truncation, info
