import json
import os, re
import numpy as np
import cv2
import pyastar2d as pyastar
import random
import time
import math
import copy
from PIL import Image
from utils.utils import extract_main_message

from LLM.LLM import LLM

CELL_SIZE = 0.125  # Grid cell size in meters
ANGLE = 15  # Default agent rotation angle


def pos2map(x, z, _scene_bounds):
    """
    Convert world coordinates (x, z) to discrete map indices.
    """
    i = int(round((x - _scene_bounds["x_min"]) / CELL_SIZE))
    j = int(round((z - _scene_bounds["z_min"]) / CELL_SIZE))
    return i, j


class lm_agent:
    def __init__(self, agent_id, logger, max_frames, args, output_dir='results'):
        # State and environment info
        self.with_oppo = None
        self.oppo_pos = None
        self.with_character = None
        self.color2id = None
        self.satisfied = None
        self.object_list = None
        self.container_held = None
        self.gt_mask = None
        self.object_info = {}  # Object metadata
        self.object_per_room = {}  # Room-wise object allocation

        # Mapping
        self.wall_map = None
        self.id_map = None
        self.object_map = None
        self.known_map = None
        self.occupancy_map = None

        # Agent config
        self.agent_id = agent_id
        self.agent_type = 'lm_agent'
        self.agent_names = ["Alice", "Bob"]
        self.opponent_agent_id = 1 - agent_id
        self.env_api = None
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.map_size = (240, 120)
        self.save_img = True
        self._scene_bounds = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
        self.max_nav_steps = 80
        self.max_move_steps = 150
        self.space_upd_freq = 30
        self.logger = logger
        random.seed(1024)
        self.debug = True

        # Perception
        self.local_occupancy_map = None
        self.new_object_list = None
        self.visible_objects = None
        self.num_frames = None
        self.steps = None
        self.obs = None
        self.local_step = 0

        # Action tracking
        self.last_action = None
        self.pre_action = None
        self.goal_objects = None
        self.dropping_object = None

        # LLM-based decision system
        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.LLM = LLM(self.source, self.lm_id, self.prompt_template_path,
                       self.communication, self.cot, self.args, self.agent_id)
        self.action_history = []
        self.dialogue_history = []
        self.plan = None
        self.progress_desc = ''

        # Navigation and interaction state
        self.rooms_name = None
        self.rooms_explored = {}
        self.position = None
        self.forward = None
        self.current_room = None
        self.holding_objects_id = None
        self.oppo_holding_objects_id = None
        self.oppo_last_room = None
        self.rotated = None
        self.navigation_threshold = 5
        self.detection_threshold = 5
        self.Alice_last_plan = "none"
        self.Bob_last_plan = "none"

    def pos2map(self, x, z):
        """Convert world (x, z) position to discrete map coordinates (i, j)."""
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j

    def map2pos(self, i, j):
        """Convert discrete map coordinates (i, j) back to world (x, z) position."""
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z

    def get_pc(self, color):
        """Extract 3D point cloud for a given color mask from the observation."""
        depth = self.obs['depth'].copy()
        # Filter out irrelevant pixels based on segmentation mask
        for i in range(len(self.obs['seg_mask'])):
            for j in range(len(self.obs['seg_mask'][0])):
                if (self.obs['seg_mask'][i][j] != color).any():
                    depth[i][j] = 1e9  # Mark as invalid

        # Camera intrinsics
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx, cy = W / 2., H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))

        # Project to camera space
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)
        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth

        index = np.where((depth > 0) & (depth < 10))
        xx = xx[index].reshape(-1)
        yy = yy[index].reshape(-1)
        depth = depth[index].reshape(-1)

        # Homogeneous coordinates for transformation
        pc = np.stack((xx, yy, depth, np.ones_like(xx))).reshape(4, -1)

        # Transform from camera to world coordinates
        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        rpc = np.dot(inv_E, pc)

        return rpc[:3]  # Return 3D points only

    def cal_object_position(self, o_dict):
        """Estimate the 3D position of an object from its segmentation mask."""
        pc = self.get_pc(o_dict['seg_color'])
        if pc.shape[1] < 5:
            return None
        position = pc.mean(1)
        return position[:3]

    def filtered(self, all_visible_objects):
        """Filter out irrelevant objects (type >= 4)."""
        visible_obj = []
        for o in all_visible_objects:
            if o['type'] is not None and o['type'] < 4:
                visible_obj.append(o)
        return visible_obj

    def get_object_list(self):
        """
        Aggregate objects in the environment and categorize them by type and room.
        Types:
            0 - target object,
            1 - container,
            2 - goal object
        """
        object_list = {0: [], 1: [], 2: []}
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}

        for object_type in [0, 1, 2]:
            obj_map_indices = np.where(self.object_map == object_type + 1)
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                obj_id = self.id_map[i, j]

                # Skip if already processed or held
                if obj_id in self.satisfied or obj_id in self.holding_objects_id or obj_id in self.oppo_holding_objects_id or \
                        self.object_info[obj_id] in object_list[object_type]:
                    continue

                object_list[object_type].append(self.object_info[obj_id])
                room = self.env_api['belongs_to_which_room'](self.object_info[obj_id]['position'])
                if room is None:
                    self.logger.warning(f"obj {self.object_info[obj_id]} not in any room")
                    continue
                self.object_per_room[room][object_type].append(self.object_info[obj_id])

        self.object_list = object_list

    def get_new_object_list(self):
        """
        Detect and categorize visible objects in the current frame.
        Object types:
            0 - target object
            1 - container
            2 - goal object
            3 - agent (opponent)
        Updates object map, id map, and records new objects.
        """
        self.visible_objects = self.obs['visible_objects']
        self.new_object_list = {0: [], 1: [], 2: []}

        for o_dict in self.visible_objects:
            if o_dict['id'] is None:
                continue
            self.color2id[o_dict['seg_color']] = o_dict['id']

            if o_dict['id'] in self.satisfied or o_dict['id'] in self.with_character or o_dict['type'] == 4:
                continue

            position = self.cal_object_position(o_dict)
            if position is None:
                continue

            object_id = o_dict['id']
            new_obj = object_id not in self.object_info

            self.object_info.setdefault(object_id, {})
            self.object_info[object_id].update({
                'id': object_id,
                'type': o_dict['type'],
                'name': o_dict['name'],
                'position': position
            })

            if o_dict['type'] == 3:  # Opponent agent
                if object_id == self.opponent_agent_id:
                    self.oppo_pos = position
                    if position is not None:
                        oppo_last_room = self.env_api['belongs_to_which_room'](position)
                        if oppo_last_room is not None:
                            self.oppo_last_room = oppo_last_room
                continue

            if object_id in self.satisfied or object_id in self.with_character:
                continue

            x, y, z = position
            i, j = self.pos2map(x, z)

            if self.object_map[i, j] == 0:
                if o_dict['type'] == 0:
                    self.object_map[i, j] = 1
                    self.new_object_list[0].append(object_id) if new_obj else None
                elif o_dict['type'] == 1:
                    self.object_map[i, j] = 2
                    self.new_object_list[1].append(object_id) if new_obj else None
                elif o_dict['type'] == 2:
                    self.object_map[i, j] = 3
                    self.new_object_list[2].append(object_id) if new_obj else None
                self.id_map[i, j] = object_id

    def color2id_fc(self, color):
        """
        Convert a segmentation color to corresponding object ID.
        Returns -100 for walls, or agent_id for the current agent.
        """
        if color not in self.color2id:
            if (color != self.agent_color).any():
                return -100
            return self.agent_id
        return self.color2id[color]

    def dep2map(self):
        """
        Project depth map to 2D occupancy and wall maps.
        Removes moved objects, updates known and unknown spaces.
        """
        local_known_map = np.zeros_like(self.occupancy_map, np.int32)
        depth = self.obs['depth']

        # Mask out characters from depth using segmentation mask
        filter_depth = depth.copy()
        for i in range(depth.shape[0]):
            for j in range(depth.shape[1]):
                if self.color2id_fc(tuple(self.obs['seg_mask'][i, j])) in self.with_character:
                    filter_depth[i, j] = 1e9
        depth = filter_depth

        # Camera intrinsics
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx, cy = W / 2., H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))

        # Back-project depth to 3D point cloud in camera frame
        x_idx, y_idx = np.meshgrid(np.linspace(0, W - 1, W), np.linspace(0, H - 1, H))
        xx = (x_idx - cx) / fx * depth
        yy = (y_idx - cy) / fy * depth
        pc = np.stack((xx, yy, depth, np.ones_like(depth))).reshape(4, -1)

        # Transform to world frame
        E = np.array(self.obs['camera_matrix']).reshape((4, 4))
        inv_E = np.linalg.inv(E)
        rot = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        rpc = (inv_E @ rot @ pc).reshape(4, -1)

        # Map coordinates to grid space
        X = np.clip(np.rint((rpc[0] - self._scene_bounds["x_min"]) / CELL_SIZE), 0, self.map_size[0] - 1)
        Z = np.clip(np.rint((rpc[2] - self._scene_bounds["z_min"]) / CELL_SIZE), 0, self.map_size[1] - 1)

        # Known space update
        idx = np.where((depth.flatten() > 0) & (depth.flatten() < self.detection_threshold) & (rpc[1] < 1.5))
        XX, ZZ = X[idx].astype(np.int32), Z[idx].astype(np.int32)
        local_known_map[XX, ZZ] = 1

        # Clear out removed objects (very low height)
        idx = np.where((depth.flatten() > 0) & (depth.flatten() < self.navigation_threshold) & (rpc[1] < 0.05))
        self.occupancy_map[X[idx].astype(np.int32), Z[idx].astype(np.int32)] = 0

        # Mark obstacles (valid height)
        idx = np.where(
            (depth.flatten() > 0) & (depth.flatten() < self.navigation_threshold) & (rpc[1] > 0.1) & (rpc[1] < 1.5))
        XX, ZZ = X[idx].astype(np.int32), Z[idx].astype(np.int32)
        self.occupancy_map[XX, ZZ] = 1
        self.local_occupancy_map[XX, ZZ] = 1

        # Mark walls (very tall obstacles)
        idx = np.where(
            (depth.flatten() > 0) & (depth.flatten() < self.navigation_threshold) & (rpc[1] > 2) & (rpc[1] < 3))
        XX, ZZ = X[idx].astype(np.int32), Z[idx].astype(np.int32)
        self.wall_map[XX, ZZ] = 1

        return local_known_map

    def l2_distance(self, st, g):
        """Euclidean distance between two 2D points"""
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5

    def get_angle(self, forward, origin, position):
        """
        Compute the signed angle (in degrees) between agent's facing direction and a target position.
        """
        p0 = np.array([origin[0], origin[2]])
        p1 = np.array([position[0], position[2]])
        d = p1 - p0
        d = d / np.linalg.norm(d)
        f = np.array([forward[0], forward[2]])

        dot = f[0] * d[0] + f[1] * d[1]
        det = f[0] * d[1] - f[1] * d[0]
        angle = np.arctan2(det, dot)
        return np.rad2deg(angle)

    def reach_target_pos(self, target_pos, threshold=1.0):
        """
        Check if the agent has reached the target position.
        If the task is 'transport', also check room consistency.
        """

        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        if self.plan.startswith('transport'):
            if self.env_api['belongs_to_which_room'](np.array([x, 0, z])) != \
               self.env_api['belongs_to_which_room'](np.array([gx, 0, gz])):
                return False
        return d < threshold

    def conv2d(self, map, kernel=3):
        """
        Apply 2D convolution to a map with uniform kernel.
        Used for spatial expansion or smoothing.
        """
        from scipy.signal import convolve2d
        conv = np.ones((kernel, kernel))
        return convolve2d(map, conv, mode='same', boundary='fill')

    def find_shortest_path(self, st, goal, map=None):
        """
        Compute shortest path on the cost-adjusted map using A*.
        Higher cost is assigned to obstacles, unknown areas, and walls.
        """
        st_x, _, st_z = st
        g_x, _, g_z = goal
        st_i, st_j = self.pos2map(st_x, st_z)
        g_i, g_j = self.pos2map(g_x, g_z)

        dist_map = np.ones_like(map, dtype=np.float32)
        super_map1 = self.conv2d(map, kernel=5)
        dist_map[super_map1 > 0] = 5
        super_map2 = self.conv2d(map)
        dist_map[super_map2 > 0] = 10
        dist_map[map > 0] = 50
        dist_map[self.known_map == 0] += 5
        dist_map[self.wall_map == 1] += 10000

        path = pyastar.astar_path(dist_map, (st_i, st_j), (g_i, g_j), allow_diagonal=False)
        return path

    def reset(self, obs, goal_objects=None, output_dir=None, env_api=None, rooms_name=None,
              agent_color=[-1, -1, -1], agent_id=0, gt_mask=True, save_img=True):
        """
        Reset the agent's internal state for a new episode.
        Initializes maps, object state, goal definitions, and LLM planning.
        """
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.agent_color = agent_color
        self.agent_id = agent_id
        self.rooms_name = rooms_name
        self.room_distance = 0
        assert type(goal_objects) == dict
        self.goal_objects = goal_objects
        self.oppo_pos = None
        goal_count = sum(goal_objects.values())
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None

        # Initialize internal maps
        self.occupancy_map = np.zeros(self.map_size, np.int32)
        self.known_map = np.zeros(self.map_size, np.int32)
        self.object_map = np.zeros(self.map_size, np.int32)
        self.id_map = np.zeros(self.map_size, np.int32)
        self.wall_map = np.zeros(self.map_size, np.int32)
        self.local_occupancy_map = np.zeros(self.map_size, np.int32)

        # Initialize memory and task-related variables
        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.new_object_list = {0: [], 1: [], 2: []}
        self.container_held = None
        self.holding_objects_id = []
        self.oppo_holding_objects_id = []
        self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
        self.satisfied = []
        self.color2id = {}
        self.dropping_object = []
        self.steps = 0
        self.num_frames = 0

        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api['belongs_to_which_room'](self.position)

        if agent_id == 0:
            self.distances_to_all_rooms = self.env_api['distances_to_all_rooms'](self.position, "Alice")
        else:
            self.distances_to_all_rooms = self.env_api['distances_to_all_rooms'](self.position, "Bob")

        self.rotated = None
        self.rooms_explored = {}
        self.plan = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        self.gt_mask = gt_mask

        # Ground truth mask or detection model
        if self.gt_mask:
            self.detection_threshold = 5
        else:
            self.detection_threshold = 3
            from detection import init_detection
            self.detection_model = init_detection()

        self.navigation_threshold = 5
        self.LLM.reset(self.rooms_name, self.goal_objects)
        self.save_img = save_img
        self.progress_desc = ''


    def move(self, target_pos):
        """Navigate towards the target position and return the next action."""
        self.local_step += 1
        local_known_map = self.dep2map()

        # Periodically update the local occupancy map
        if self.local_step % self.space_upd_freq == 0:
            print("update local map")
            self.local_occupancy_map = copy.deepcopy(self.occupancy_map)

        # Merge local knowledge into global known map
        self.known_map = np.maximum(self.known_map, local_known_map)

        # Plan path and determine intermediate waypoint
        path = self.find_shortest_path(self.position, target_pos, self.local_occupancy_map)
        i, j = path[min(5, len(path) - 1)]
        x, z = self.map2pos(i, j)

        # Determine rotation or movement action based on angle
        angle = self.get_angle(forward=np.array(self.forward),
                               origin=np.array(self.position),
                               position=np.array([x, 0, z]))
        if np.abs(angle) < ANGLE:
            action = {"type": 0}  # move forward
        elif angle > 0:
            action = {"type": 1}  # rotate left
        else:
            action = {"type": 2}  # rotate right
        return action

    def draw_map(self, previous_name):
        """Render the agent's internal map and save it as an image."""
        draw_map = np.zeros((self.map_size[0], self.map_size[1], 3))

        # Assign colors for known, unknown, wall, and object types
        for i in range(self.map_size[0]):
            for j in range(self.map_size[1]):
                if self.occupancy_map[i, j] > 0:
                    draw_map[i, j] = 100
                if self.known_map[i, j] == 0:
                    draw_map[i, j] = 50
                if self.wall_map[i, j] > 0:
                    draw_map[i, j] = 150

        draw_map[np.where(self.object_map == 1)] = [255, 0, 0]    # object type 1
        draw_map[np.where(self.object_map == 2)] = [0, 255, 0]    # object type 2
        draw_map[np.where(self.object_map == 3)] = [0, 0, 255]    # object type 3

        # Draw agent and opponent positions
        if self.oppo_pos is not None:
            draw_map[self.pos2map(self.oppo_pos[0], self.oppo_pos[2])] = [0, 255, 255]
        draw_map[self.pos2map(self.obs["agent"][0], self.obs["agent"][2])] = [255, 255, 0]

        draw_map = np.rot90(draw_map, 1)  # Rotate for visual alignment
        cv2.imwrite(previous_name + '_map.png', draw_map)

    def gotoroom(self):
        """Move to the specified room if not already there."""
        target_room = ' '.join(self.plan.split(' ')[2:4]).rstrip(',')
        print(target_room)
        target_pos = self.env_api['center_of_room'](target_room)

        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None

        # Interrupt plan if new objects are observed
        if len(self.new_object_list[0]) + len(self.new_object_list[1]) + len(self.new_object_list[2]) > 0:
            self.action_history[-1] = self.action_history[-1].replace(self.plan, f'go to {self.current_room}')
            self.new_object_list = {0: [], 1: [], 2: []}
            self.plan = None
            return None

        return self.move(target_pos)

    def goexplore(self):
        """Explore current room by rotating and moving until all areas are observed."""
        target_room = ' '.join(self.plan.split(' ')[-2:])
        target_pos = self.env_api['center_of_room'](target_room)
        self.explore_count += 1
        dis_threshold = 1 + self.explore_count / 50

        # Navigate until reaching target position
        if not self.reach_target_pos(target_pos, dis_threshold):
            return self.move(target_pos)

        # Perform in-place rotation to explore surroundings
        if self.rotated is None:
            self.rotated = 0
        if self.rotated == 16:
            self.rotated = 0
            self.rooms_explored[target_room] = 'all'
            self.plan = None
            return None
        self.rotated += 1
        return {"type": 1}  # rotate left

    def gograsp(self):
        """Move to and attempt to grasp the specified object."""
        target_object_id = int(self.plan.split(' ')[-1][1:-1])

        # If already held, mark success
        if target_object_id in self.holding_objects_id:
            self.logger.info(f"successful holding!")
            self.object_map[np.where(self.id_map == target_object_id)] = 0
            self.id_map[np.where(self.id_map == target_object_id)] = 0
            self.plan = None
            return None

        # Initialize target position
        if self.target_pos is None:
            print(self.object_info[target_object_id])
            self.target_pos = copy.deepcopy(self.object_info[target_object_id]['position'])
        target_object_pos = self.target_pos

        # Abort if object is no longer available
        if target_object_id not in self.object_info or target_object_id in self.with_oppo:
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            return None

        # Navigate to target object
        if not self.reach_target_pos(target_object_pos):
            return self.move(target_object_pos)

        # Perform grasp action
        action = {
            "type": 3,
            "object": target_object_id,
            "arm": 'left' if self.obs["held_objects"][0]['id'] is None else 'right'
        }
        return action


    def goput(self):
        """Navigate to the drop position and drop held object(s)."""
        if len(self.holding_objects_id) == 0:
            # No object is being held; cancel plan
            self.plan = None
            self.with_character = [self.agent_id]
            return None

        if self.target_pos is None:
            # Assign drop target position from predefined object list
            self.target_pos = copy.deepcopy(self.object_list[2][0]['position'])
        target_pos = self.target_pos

        # If not yet at target position, keep navigating
        if not self.reach_target_pos(target_pos, 1.5):
            return self.move(target_pos)

        # Drop the object based on which arm is holding it
        if self.obs["held_objects"][0]['type'] is not None:
            self.dropping_object += [self.obs["held_objects"][0]['id']]
            if self.obs["held_objects"][0]['type'] == 1:  # Container object
                self.dropping_object += [
                    x for x in self.obs["held_objects"][0]['contained'] if x is not None
                ]
            return {"type": 5, "arm": "left"}
        else:
            self.dropping_object += [self.obs["held_objects"][1]['id']]
            if self.obs["held_objects"][1]['type'] == 1:
                self.dropping_object += [
                    x for x in self.obs["held_objects"][1]['contained'] if x is not None
                ]
            return {"type": 5, "arm": "right"}

    def putin(self):
        """Execute action to put an object into a container (if exactly one object held)."""
        if len(self.holding_objects_id) == 1:
            self.logger.info("Successful putin")
            self.plan = None
            return None
        return {"type": 4}  # Action type 4: Put-in

    def detect(self):
        """Run object detection and return recognized objects and segmentation mask."""
        detect_result = self.detection_model(self.obs['rgb'][..., [2, 1, 0]])['predictions'][0]
        obj_infos = []
        curr_seg_mask = np.zeros((self.obs['rgb'].shape[0], self.obs['rgb'].shape[1], 3), dtype=np.int32)
        curr_seg_mask.fill(-1)

        for i in range(len(detect_result['labels'])):
            if detect_result['scores'][i] < 0.3:
                continue  # Filter low-confidence detections
            mask = detect_result['masks'][:, :, i]
            label = detect_result['labels'][i]
            curr_info = self.env_api['get_id_from_mask'](
                mask=mask,
                name=self.detection_model.cls_to_name_map(label)
            ).copy()
            if curr_info['id'] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info['seg_color']

        # Overlay character masks if present
        curr_with_seg, curr_seg_flag = self.env_api['get_with_character_mask'](
            character_object_ids=self.with_character
        )
        curr_seg_mask = (
            curr_seg_mask * (~np.expand_dims(curr_seg_flag, axis=-1)) +
            curr_with_seg * np.expand_dims(curr_seg_flag, axis=-1)
        )
        return obj_infos, curr_seg_mask

    def LLM_envs(self):
        """Invoke LLM to analyze current environment state and context."""
        return self.LLM.run_envs(
            self.num_frames,
            self.current_room,
            self.rooms_explored,
            self.obs['held_objects'],
            [self.object_info[x] for x in self.satisfied if x in self.object_info],
            self.object_list,
            self.object_per_room,
            self.action_history,
            self.obs['oppo_held_objects'],
            self.oppo_last_room,
            self.distances_to_all_rooms
        )

    def LLM_mcts(self, Alice_progress_desc, Bob_progress_desc, last_action):
        """Run LLM-guided MCTS (Monte Carlo Tree Search) based on progress and last action."""
        return self.LLM.run_mcts(self.dialogue_history, Alice_progress_desc, Bob_progress_desc, last_action)

    def LLM_get_plan(self, action_plan):
        """Query LLM to interpret or refine an action plan."""
        return self.LLM.run_get_plan(action_plan)

    def LLM_plan(self, last_plan):
        """Use LLM to generate a new plan based on previous one."""
        return self.LLM.run_planning(last_plan)

    def preact(self, obs):
        # Update internal observation state
        self.obs = obs.copy()
        self.obs['rgb'] = self.obs['rgb'].transpose(1, 2, 0)  # Convert image to HWC format
        self.num_frames = obs['current_frames']
        self.steps += 1

        # Detect objects from visual input if no ground-truth mask is provided
        if not self.gt_mask:
            self.obs['visible_objects'], self.obs['seg_mask'] = self.detect()

        # Handle invalid actions
        if obs['valid'] == False:
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.satisfied.append(self.last_action['object'])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 10, "invalid action for 10 times"

        # Update agent's position and orientation
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]

        # Query current room location
        current_room = self.env_api['belongs_to_which_room'](self.position)
        if current_room is not None:
            self.current_room = current_room

        # Estimate room distance
        self.room_distance = self.env_api['get_room_distance'](self.position)

        # Update exploration status
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            self.rooms_explored[self.current_room] = 'part'
        if self.agent_id not in self.with_character:
            self.with_character.append(self.agent_id)  # TEMP: workaround for environment bug

        # Process held objects (self)
        self.holding_objects_id = []
        self.with_oppo = []
        self.oppo_holding_objects_id = []
        for x in self.obs['held_objects']:
            self.holding_objects_id.append(x['id'])
            if x['id'] not in self.with_character:
                self.with_character.append(x['id'])
            if x['type'] == 1:  # If container, add contained items
                for y in x['contained']:
                    if y is not None and y not in self.with_character:
                        self.with_character.append(y)

        # Process opponent-held objects
        oppo_name = {}
        oppo_type = {}
        oppo_holding_objects = []
        for x in self.obs['oppo_held_objects']:
            self.oppo_holding_objects_id.append(x['id'])
            self.with_oppo.append(x['id'])
            oppo_name[x['id']] = x['name']
            oppo_type[x['id']] = x['type']
            oppo_holding_objects.append(f'<{x["name"]}>({x["id"]})')
            if x['type'] == 1:  # If container, include contents
                for i, y in enumerate(x['contained']):
                    if y is not None:
                        self.with_oppo.append(y)
                        oppo_name[y] = x['contained_name'][i]
                        oppo_type[y] = 0
                        oppo_holding_objects.append(f'<{x["contained_name"][i]}>({y})')

        # Remove opponent-held objects from map
        for obj in self.with_oppo:
            if obj not in self.satisfied:
                self.satisfied.append(obj)
                self.object_info[obj] = {
                    "name": oppo_name[obj],
                    "id": obj,
                    "type": oppo_type[obj],
                }
                self.object_map[np.where(self.id_map == obj)] = 0
                self.id_map[np.where(self.id_map == obj)] = 0

        # Remove failed interaction object
        if not self.obs['valid']:
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0

        # Successful object drop
        if len(self.dropping_object) > 0 and self.obs['status'] == 1:
            self.logger.info(f"Drop object: {self.dropping_object}")
            self.satisfied += self.dropping_object
            self.dropping_object = []
            if len(self.holding_objects_id) == 0:
                self.logger.info("successful drop!")
                self.plan = None

        # Update visible object lists
        self.get_new_object_list()
        self.get_object_list()

        # Save current occupancy map for visualization
        if self.save_img:
            self.draw_map(previous_name=f'{self.output_dir}/Images/{self.agent_id}/{self.steps:04}')

        # Update LLM environment description
        self.progress_desc = self.LLM_envs()
        empty_new = all(len(lst) == 0 for lst in self.new_object_list.values())

        # Decide whether to trigger communication
        if self.plan is None:
            return True, f'{self.agent_names[self.agent_id]} maybe has just finished an action or start to work', self.progress_desc
        if not empty_new:
            return True, f'{self.agent_names[self.agent_id]} found new objects: {self.new_object_list}', self.progress_desc

        return False, 'No changes', self.progress_desc

    def mcts(self, Alice_progress_desc, Bob_progress_desc):
        # Construct last action summary for context
        last_action = "Alice: " + self.Alice_last_plan + "   Bob: " + self.Bob_last_plan
        self.last_action = last_action

        # Generate plan via MCTS-enhanced LLM reasoning
        action_plan, new_dialogue_history = self.LLM_mcts(Alice_progress_desc, Bob_progress_desc, last_action)

        # Maintain recent dialogue history (limit to 4)
        self.dialogue_history += new_dialogue_history
        if len(self.dialogue_history) > 4:
            self.dialogue_history = self.dialogue_history[-4:]

        return action_plan

    def get_plan(self, action_plan):
        # Parse structured plan from raw LLM output
        action_plan = self.LLM_get_plan(action_plan)
        return action_plan

    def act(self, agent, last_plan):
        action = None

        lm_times = 0
        while action is None:
            if self.plan is None:  # generate plan when compelte an plan or no plan
                self.target_pos = None
                if lm_times > 3:
                    raise Exception(f"retrying LM_plan too many times")
                #  generate plan from LLM
                plan = self.LLM_plan(last_plan)
                if plan is None:  # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"
                self.plan = plan
                if agent == "Alice":
                    self.Alice_last_plan = plan
                    print("alice {}".format(self.Alice_last_plan))
                else:
                    self.Bob_last_plan = plan
                    print("bob {}".format(self.Bob_last_plan))

                lm_times += 1

            # action is still ongoing
            if self.obs['status'] == 0:
                if agent == "Alice":
                    return {'type': 'ongoing'}, self.Alice_last_plan
                else:
                    return {'type': 'ongoing'}, self.Bob_last_plan

            # execute the high-level plan
            if self.plan.startswith('go to'):
                action = self.gotoroom()
            elif self.plan.startswith('explore'):
                self.explore_count = 0
                action = self.goexplore()
            elif self.plan.startswith('go grasp'):
                action = self.gograsp()
            elif self.plan.startswith('put'):
                action = self.putin()
            elif self.plan.startswith('transport'):
                action = self.goput()
            elif self.plan.startswith('wait'):
                action = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        self.logger.info(f'{self.agent_names[self.agent_id]}: {self.plan}')

        self.last_action = action
        if agent == "Alice":
            # print("alice {}".format(self.Alice_last_plan))
            return action, self.Alice_last_plan
        else:
            # print("bob {}".format(self.Bob_last_plan))
            return action, self.Bob_last_plan
