from os import path
from typing import Optional

import numpy as np

from boxes_for_space import Box
import cv2
import utils

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


class StarshipEnv():
    """
       ### Description
    Inspired by the code from the inverted pendulum in OpenAI's gym environment.

    The inverted pendulum swingup problem is based on the classic problem in control theory.
    The system consists of a pendulum attached at one end to a fixed point, and the other end being free.
    The pendulum starts in a random position and the goal is to apply torque on the free end to swing it
    into an upright position, with its center of gravity right above the fixed point.

    The diagram below specifies the coordinate system used for the implementation of the pendulum's
    dynamic equations.

    ![Pendulum Coordinate System](./diagrams/pendulum.png)

    -  `x-y`: cartesian coordinates of the pendulum's end in meters.
    - `theta` : angle in radians.
    - `tau`: torque in `N m`. Defined as positive _counter-clockwise_.

    ### Action Space

    The action is a `ndarray` with shape `(2,)` representing the thrust and the gimbal angle of the engine.

    ### Observation Space

    UPDATE OBSERVATION SPACE DOCS HERE

    ### Rewards

    The reward function is defined as:

    UPDATE REWARD FUNCTION INFORMATION HERE

    ### Episode Truncation

    The episode truncates at 200 time steps.

    ### Arguments

    - `g`: acceleration of gravity measured in *(m s<sup>-2</sup>)* used to calculate the pendulum dynamics.
      The default value is g = 10.0 .

    ```
    gym.make('Pendulum-v1', g=9.81)
    ```

    ### Version History

    * v1: Simplify the math equations, no difference in behavior.
    * v0: Initial versions release (1.0.0)

    """
    def __init__(self, path_to_bg_img=None):

        self.g = 9.8
        self.H = 50  # rocket height (meters)
        self.I = 1/12*self.H*self.H  # Moment of inertia
        self.dt = 0.05

        self.world_x_min = -300  # meters
        self.world_x_max = 300
        self.world_y_min = -30
        self.world_y_max = 570
        self.max_thrust = 20.0 # N?
        self.max_gimbal  = 30 * np.pi/180.0 # radians?
        self.max_steps = 800

        # target point
        self.target_x, self.target_y, self.target_r = 0, self.H/2.0, 50

        self.already_landing = False
        self.already_crash = False

        # viewport height x width (pixels)
        self.viewport_h = int(768)
        self.viewport_w = int(768 * (self.world_x_max-self.world_x_min) \
                          / (self.world_y_max - self.world_y_min))
        self.step_id = 0

        if path_to_bg_img is None:
            path_to_bg_img = 'landing.jpg'
        self.bg_img = utils.load_bg_img(path_to_bg_img, w=self.viewport_w, h=self.viewport_h)

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = Box(
            low=np.array([0.0, -self.max_gimbal]), high=np.array([self.max_thrust, self.max_gimbal]), shape=(2,), dtype=np.float32)
        # For now setting no bounds on the state observations, might need to change in the future
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = Box(low=-high, high=high, dtype=np.float32)

        self.state_buffer = []
        self.action_buffer = []

    # Created
    _np_random: Optional[np.random.Generator] = None

    @property
    def np_random(self) -> np.random.Generator:
        """Returns the environment's internal :attr:`_np_random` that if not set will initialise with a random seed."""
        if self._np_random is None:
            self._np_random, seed = utils.np_random()
        return self._np_random

    @np_random.setter
    def np_random(self, value: np.random.Generator):
        self._np_random = value

    def step(self, u):
        x, y, vx, vy = self.state[0], self.state[1], self.state[2], self.state[3]
        theta, vtheta = self.state[4], self.state[5]

        f, phi = u[0], u[1]
        f = np.clip(f, -self.max_thrust, self.max_thrust)
        phi = np.clip(phi, -self.max_gimbal, self.max_gimbal)
        self.last_u = u

        ft, fr = -f*np.sin(phi), f*np.cos(phi)
        fx = ft*np.cos(theta) - fr*np.sin(theta)
        fy = ft*np.sin(theta) + fr*np.cos(theta)

        rho = 1 / (125/(self.g/2.0))**0.5  # suppose after 125 m free fall, then air resistance = mg
        ax, ay = fx-rho*vx, fy-self.g-rho*vy
        atheta = ft*self.H/2 / self.I


        # update agent
        if self.already_landing:
            vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
            phi, f = 0, 0

        x_new = x + vx*self.dt + 0.5 * ax * (self.dt**2)
        y_new = y + vy*self.dt + 0.5 * ay * (self.dt**2)
        vx_new, vy_new = vx + ax * self.dt, vy + ay * self.dt
        theta_new = theta + vtheta*self.dt + 0.5 * atheta * (self.dt**2)
        vtheta_new = vtheta + atheta * self.dt

        self.step_id += 1
        self.state = np.array([x_new, y_new, vx_new, vy_new, theta_new, vtheta_new])

        # Cost is currently taken to the -ve of the reward from the 
        rewards = self.calculate_reward(self.state)

        self.already_landing = self.check_landing_success(self.state)
        self.already_crash = self.check_crash(self.state)

        self.state_buffer.append(self.state)
        self.action_buffer.append(u)

        if self.already_crash or self.already_landing:
            done = True
        else:
            done = False

        return self._get_obs(), rewards, done, False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        self.step_id = 0
        self.already_landing = False
        cv2.destroyAllWindows()

        if seed is not None:
            self._np_random, seed = utils.np_random(seed)
        if options is None:
            state = self.create_random_state()
        else:
            # Note that if you use custom reset bounds, it may lead to out-of-bound
            # state/observations.
            x = options.get("x_init") if "x_init" in options else DEFAULT_X
            y = options.get("y_init") if "y_init" in options else DEFAULT_Y
            x = utils.verify_number_and_cast(x)
            y = utils.verify_number_and_cast(y)
            state = np.array([x, y, 0.0, 0.0, 0.0, 0.0])
        self.state = state
        self.state_buffer = []
        self.action_buffer = []
        self.last_u = np.zeros(2)

        return self._get_obs()

    def _get_obs(self):
        return self.state

    def create_random_state(self):

        # predefined locations
        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min
        xc = (self.world_x_max + self.world_x_min) / 2.0
        yc = (self.world_y_max + self.world_y_min) / 2.0

        x = np.random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
        y = yc + 0.4*y_range
        if x <= 0:
            theta = -85 / 180 * np.pi
        else:
            theta = 85 / 180 * np.pi
        vy = -50

        state = np.array([x, y, 0, vy, theta, 0.0])
        return state

    def check_crash(self, state):
        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        theta = state[4]
        vtheta = state[5]
        v = (vx**2 + vy**2)**0.5

        crash = False
        if y >= self.world_y_max - self.H / 2.0:
            crash = True
        if y <= 0 + self.H / 2.0 and v >= 15.0:
            crash = True
        if y <= 0 + self.H / 2.0 and abs(x) >= self.target_r:
            crash = True
        if y <= 0 + self.H / 2.0 and abs(theta) >= 10/180*np.pi:
            crash = True
        if y <= 0 + self.H / 2.0 and abs(vtheta) >= 10/180*np.pi:
            crash = True
        return crash

    def check_landing_success(self, state):
        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        theta = state[4]
        vtheta = state[5]
        v = (vx**2 + vy**2)**0.5
        return True if y <= 0 + self.H / 2.0 and v < 15.0 and abs(x) < self.target_r \
                        and abs(theta) < 10/180*np.pi and abs(vtheta) < 10/180*np.pi else False

    def render(self, window_name='env', wait_time=1,
               with_trajectory=True, with_camera_tracking=True,
               crop_scale=0.4):

        canvas = np.copy(self.bg_img)
        polys = self.create_polygons()

        # draw target region
        for poly in polys['target_region']:
            self.draw_a_polygon(canvas, poly)
        # draw rocket
        for poly in polys['rocket']:
            self.draw_a_polygon(canvas, poly)
        frame_0 = canvas.copy()

        # draw engine work
        for poly in polys['engine_work']:
            self.draw_a_polygon(canvas, poly)
        frame_1 = canvas.copy()

        if with_camera_tracking:
            frame_0 = self.crop_alongwith_camera(frame_0, crop_scale=crop_scale)
            frame_1 = self.crop_alongwith_camera(frame_1, crop_scale=crop_scale)

        # draw trajectory
        if with_trajectory:
            self.draw_trajectory(frame_0)
            self.draw_trajectory(frame_1)

        # draw text
        self.draw_text(frame_0, color=(0, 0, 0))
        self.draw_text(frame_1, color=(0, 0, 0))

        cv2.imshow(window_name, frame_0[:,:,::-1])
        cv2.waitKey(wait_time)
        cv2.imshow(window_name, frame_1[:,:,::-1])
        cv2.waitKey(wait_time)
        return frame_0, frame_1
    
    def calculate_reward(self, state):

        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min

        # dist between agent and target point
        dist_x = abs(state[0] - self.target_x)
        dist_y = abs(state[1] - self.target_y)
        dist_norm = dist_x / x_range + dist_y / y_range

        dist_reward = 0.25*(1.0 - dist_norm)

        if abs(state[4]) <= np.pi / 6.0:
            pose_reward = 0.1
        else:
            pose_reward = 3.* abs(state[4]) / (np.pi)
            pose_reward = 0.1 * (1.0 - pose_reward)

        reward = dist_reward + pose_reward

        v = (state[2] ** 2 + state[3] ** 2) ** 0.5
        if self.already_crash:
            reward = (reward + 5*np.exp(-1*v/10.)) * (self.max_steps - self.step_id)
        if self.already_landing:
            reward = (1.0 + 5*np.exp(-1*v/10.))*(self.max_steps - self.step_id)

        return reward

    def create_polygons(self):

        polys = {'rocket': [], 'engine_work': [], 'target_region': []}

        H, W = self.H, self.H / 2.6
        dl = self.H / 30

        # rocket main body (right half)
        pts = np.array([[ 0.        ,  0.5006878 ],
                        [ 0.03125   ,  0.49243465],
                        [ 0.0625    ,  0.48143053],
                        [ 0.11458334,  0.43878955],
                        [ 0.15277778,  0.3933975 ],
                        [ 0.2326389 ,  0.23796424],
                        [ 0.2326389 , -0.49931225],
                        [ 0.        , -0.49931225]], dtype=np.float32)
        pts[:, 0] = pts[:, 0] * W
        pts[:, 1] = pts[:, 1] * H
        polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})

        # rocket main body (left half)
        pts = np.array([[-0.        ,  0.5006878 ],
                        [-0.03125   ,  0.49243465],
                        [-0.0625    ,  0.48143053],
                        [-0.11458334,  0.43878955],
                        [-0.15277778,  0.3933975 ],
                        [-0.2326389 ,  0.23796424],
                        [-0.2326389 , -0.49931225],
                        [-0.        , -0.49931225]], dtype=np.float32)
        pts[:, 0] = pts[:, 0] * W
        pts[:, 1] = pts[:, 1] * H
        polys['rocket'].append({'pts': pts, 'face_color': (212, 212, 232), 'edge_color': None})

        # upper wing (right)
        pts = np.array([[0.15972222, 0.3933975 ],
                        [0.3784722 , 0.303989  ],
                        [0.3784722 , 0.2352132 ],
                        [0.22916667, 0.23658872]], dtype=np.float32)
        pts[:, 0] = pts[:, 0] * W
        pts[:, 1] = pts[:, 1] * H
        polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

        # upper wing (left)
        pts = np.array([[-0.15972222,  0.3933975 ],
                        [-0.3784722 ,  0.303989  ],
                        [-0.3784722 ,  0.2352132 ],
                        [-0.22916667,  0.23658872]], dtype=np.float32)
        pts[:, 0] = pts[:, 0] * W
        pts[:, 1] = pts[:, 1] * H
        polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

        # lower wing (right)
        pts = np.array([[ 0.2326389 , -0.16368638],
                        [ 0.4548611 , -0.33562586],
                        [ 0.4548611 , -0.48555708],
                        [ 0.2638889 , -0.48555708]], dtype=np.float32)
        pts[:, 0] = pts[:, 0] * W
        pts[:, 1] = pts[:, 1] * H
        polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

        # lower wing (left)
        pts = np.array([[-0.2326389 , -0.16368638],
                        [-0.4548611 , -0.33562586],
                        [-0.4548611 , -0.48555708],
                        [-0.2638889 , -0.48555708]], dtype=np.float32)
        pts[:, 0] = pts[:, 0] * W
        pts[:, 1] = pts[:, 1] * H
        polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

        # engine work
        f, phi = self.last_u[0], self.last_u[1]
        c, s = np.cos(phi), np.sin(phi)

        if f > 0 and f < 0.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
        elif f > 0.5 * self.g and f < 1.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
        elif f > 1.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            pts4 = utils.create_rectangle_poly(center=(12 * dl * s, -H / 2 - 12 * dl * c), w=3 * dl, h=3 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (255, 255, 255), 'edge_color': None})
            polys['engine_work'].append({'pts': pts4, 'face_color': (255, 255, 255), 'edge_color': None})
        
        pts1 = utils.create_ellipse_poly(center=(0, 0), rx=self.target_r, ry=self.target_r/4.0)
        pts2 = utils.create_rectangle_poly(center=(0, 0), w=self.target_r/3.0, h=0)
        pts3 = utils.create_rectangle_poly(center=(0, 0), w=0, h=self.target_r/6.0)
        polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
        polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
        polys['target_region'].append({'pts': pts3, 'face_color': None, 'edge_color': (242, 242, 242)})

        # apply transformation
        for poly in polys['rocket'] + polys['engine_work']:
            M = utils.create_pose_matrix(tx=self.state[0], ty=self.state[1], rz=self.state[4])
            pts = np.array(poly['pts'])
            pts = np.concatenate([pts, np.ones_like(pts)], axis=-1)  # attach z=1, w=1
            pts = np.matmul(M, pts.T).T
            poly['pts'] = pts[:, 0:2]

        return polys


    def draw_a_polygon(self, canvas, poly):

        pts, face_color, edge_color = poly['pts'], poly['face_color'], poly['edge_color']
        pts_px = self.wd2pxl(pts)
        if face_color is not None:
            cv2.fillPoly(canvas, [pts_px], color=face_color, lineType=cv2.LINE_AA)
        if edge_color is not None:
            cv2.polylines(canvas, [pts_px], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA)

        return canvas


    def wd2pxl(self, pts, to_int=True):

        pts_px = np.zeros_like(pts)

        scale = self.viewport_w / (self.world_x_max - self.world_x_min)
        for i in range(len(pts)):
            pt = pts[i]
            x_p = (pt[0] - self.world_x_min) * scale
            y_p = (pt[1] - self.world_y_min) * scale
            y_p = self.viewport_h - y_p
            pts_px[i] = [x_p, y_p]

        if to_int:
            return pts_px.astype(int)
        else:
            return pts_px

    def draw_text(self, canvas, color=(255, 255, 0)):

        def put_text(vis, text, pt):
            cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

        pt = (10, 20)
        text = "simulation time: %.2fs" % (self.step_id * self.dt)
        put_text(canvas, text, pt)

        pt = (10, 40)
        text = "simulation steps: %d" % (self.step_id)
        put_text(canvas, text, pt)

        pt = (10, 60)
        text = "x: %.2f m, y: %.2f m" % \
               (self.state[0], self.state[1])
        put_text(canvas, text, pt)

        pt = (10, 80)
        text = "vx: %.2f m/s, vy: %.2f m/s" % \
               (self.state[2], self.state[3])
        put_text(canvas, text, pt)

        pt = (10, 100)
        text = "a: %.2f degree, va: %.2f degree/s" % \
               (self.state[4] * 180 / np.pi, self.state[5] * 180 / np.pi)
        put_text(canvas, text, pt)


    def draw_trajectory(self, canvas, color=(255, 0, 0)):

        pannel_w, pannel_h = 256, 256
        traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

        sw, sh = pannel_w/self.viewport_w, pannel_h/self.viewport_h  # scale factors

        # draw horizon line
        range_x, range_y = self.world_x_max - self.world_x_min, self.world_y_max - self.world_y_min
        pts = [[self.world_x_min + range_x/3, self.H/2], [self.world_x_max - range_x/3, self.H/2]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # draw vertical line
        pts = [[0, self.H/2], [0, self.H/2+range_y/20]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        if len(self.state_buffer) < 2:
            return

        # draw traj
        pts = []
        for state in self.state_buffer:
            pts.append([state[0], state[1]])
        pts_px = self.wd2pxl(pts)

        dn = 5
        for i in range(0, len(pts_px)-dn, dn):

            x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
            x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)

            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

        roi_x1, roi_x2 = self.viewport_w - 10 - pannel_w, self.viewport_w - 10
        roi_y1, roi_y2 = 10, 10 + pannel_h
        canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = 0.6*canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4*traj_pannel


    def crop_alongwith_camera(self, vis, crop_scale=0.4):
        x, y = self.state[0], self.state[1]
        xp, yp = self.wd2pxl([[x, y]])[0]
        crop_w_half, crop_h_half = int(self.viewport_w*crop_scale), int(self.viewport_h*crop_scale)
        # check boundary
        if xp <= crop_w_half + 1:
            xp = crop_w_half + 1
        if xp >= self.viewport_w - crop_w_half - 1:
            xp = self.viewport_w - crop_w_half - 1
        if yp <= crop_h_half + 1:
            yp = crop_h_half + 1
        if yp >= self.viewport_h - crop_h_half - 1:
            yp = self.viewport_h - crop_h_half - 1

        x1, x2, y1, y2 = xp-crop_w_half, xp+crop_w_half, yp-crop_h_half, yp+crop_h_half
        vis = vis[y1:y2, x1:x2, :]

        vis = cv2.resize(vis, (self.viewport_w, self.viewport_h))
        return vis

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
