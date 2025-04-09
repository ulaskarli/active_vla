import datetime
import glob
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tyro

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.zmq_core.camera_node import ZMQClientCamera

import dataclasses
import enum
import logging
from openpi_client import websocket_client_policy as _websocket_client_policy
from typing import List, Optional  # noqa: UP035

from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override
import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from PIL import Image

# Imports for shared control
import pygame

@dataclass
class Args:
    agent: str = "none"
    robot_port: int = 6001
    wrist_camera_port: int = 5000
    base_camera_port: int = 5001
    hostname: str = "127.0.0.1"
    hz: int = 0
    start_joints: Optional[Tuple[float, ...]] = None

    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
    prompt: str = "lift the corn"

class IQRLEnvironment(_environment.Environment):
    """An environment for an Xarm robot on real hardware in IQRL setting."""

    def __init__(
        self,
        robot_env,
        args,
        shared_control=False
    ) -> None:
        self._env = robot_env
        self.args=args
        self._ts = None
        self.home = [-0.035, -0.73, 0.042, 0.93, 0.017, 0.915, 0.0, 0.0]
        self.shared_control = shared_control
        self.controller = None
        self.uncertainty = None

        if self.shared_control:
            # Initialize Controller
            self.controller = self.init_controller()
    
    @override
    def init_controller(self):
        # Initialize Pygame
        code = pygame.init()
        print("[Shared Control] pygame initialized with code", code)

        # Initialize Xbox controller
        code = pygame.joystick.init()
        print("[Shared Control] joystick initialized with code", code)

        if pygame.joystick.get_count() < 1:
            print("[Shared Control] No controller found!")
            exit()

        controller = pygame.joystick.Joystick(0)
        controller.init()

        return controller

    @override
    def reset(self) -> None:
        print("env reset")
        self._ts = self._env.get_obs()
        current_pos = np.array(self._ts['joint_positions'])

        num_steps = 30  # Number of interpolation steps
        for t in range(1, num_steps + 1):
            interpolated_pos = current_pos + (self.home - current_pos) * (t / num_steps)
            self._ts = self._env.step(interpolated_pos)

        print("Final joint positions:", self._ts['joint_positions'])


    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ts is None:
            raise RuntimeError("Timestep is not set. Call reset() first.")
        
        obs = self._ts
        joints = obs["joint_positions"]
        wrist_image = obs["wrist_rgb"]
        base_image = obs["base_rgb"]

        # print(self.args.prompt)

        # print("-------------------JOINTS-----------------------------")
        # print(joints)

        # # Create a PIL image from the array
        # image = Image.fromarray(base_image)

        # # Display the image
        # image.show()
        return {
            "observation/state": joints,
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "prompt": self.args.prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        # print("-----------------ACTIONS-------------------------------")
        # print(action["actions"])
        
        if self.shared_control:
            if self._ts is None:
                raise RuntimeError("Timestep is not set. Call reset() first.")
            current_joint_pos = self._ts["joint_positions"]
            
            # Get xbox controller inputs
            # TODO: Potentially move this out so that we can get inputs more often (without being constrained by the VLA inference loop)
            pygame.event.pump()  # Refresh pygame events
            # Subtracting current gripper pos to make gripper input into delta
            input_velocity = self.get_input() - np.array([0, 0, 0, current_joint_pos[-1]])
            
            # Get VLA output
            next_joint_pos = self._env.step(action["actions"])
            # TODO: set up self._env so that it returns uncertainty
            uncertainty = self._env.step(action["uncertainty"])
            vla_joint_velocity = next_joint_pos - current_joint_pos
            
            # Create the combined control 
            self._ts = self.combine_control(current_joint_pos, input_velocity, vla_joint_velocity, uncertainty)
        else:
            self._ts = self._env.step(action["actions"])
    
    @override
    def get_input(self):
        # Sanity check
        assert self.controller is not None
        
        # TODO: need to check if axis 2 is what we want for z and axis 1 for gripper
        axis_x = self.controller.get_axis(4)  # Horizontal (X direction)
        axis_y = self.controller.get_axis(3)  # Vertical (Y direction)
        axis_z = self.controller.get_axis(2)  # Up & Down (Z direction)
        gripper = self.controller.get_axis(1)

        # Deadzone to avoid small movements when stick is near center
        deadzone = 0.05

        velocity_x = 0
        velocity_y = 0
        velocity_z = 0

        # Apply deadzone and get xyz movement 
        # TODO: change controller sensitivity to match VLA output as needed (default is 1)
        controller_sensitivity = 1
        if abs(axis_x) > deadzone:
            velocity_x = axis_x * controller_sensitivity

        if abs(axis_y) > deadzone:
            velocity_y = axis_y * controller_sensitivity
            
        if abs(axis_z) > deadzone:
            velocity_z = axis_z * controller_sensitivity
        
        return np.array([velocity_x, velocity_y, velocity_z, gripper])

    @override
    def combine_control(self, current_joint_pos, input_velocity, vla_joint_velocity, alpha):
        # Appending rpy to controller input velocity to match VLA output
        input_velocity_temp = np.append(input_velocity[:-1], [0, 0, 0])

        # Convert vla_joint_velocity to vla_velocity in cartesian space
        vla_velocity = self.arm.get_forward_kinematics(vla_joint_velocity[:-1], input_is_radian=True, return_is_radian=True) - self.arm.get_forward_kinematics(current_joint_pos[:-1], input_is_radian=True, return_is_radian=True)
        
        # Combine two velocities in cartesian space
        combined_velocity = vla_velocity * (1 - alpha) + input_velocity_temp * alpha
        combined_joint_pos = self.arm.get_inverse_kinematics(combined_velocity + self.arm.get_position(self, is_radian=True), input_is_radian=True, return_is_radian=True)
        
        # Combine and add back gripper movement
        combined_joint_pos = np.append(combined_joint_pos, vla_joint_velocity[-1] * (1 - alpha) + input_velocity[-1] * alpha)
        
        # Convert velocity into next pose 
        return combined_joint_pos
            


def main(args):
    camera_clients = {
            # you can optionally add camera nodes here for imitation learning purposes
            "wrist": ZMQClientCamera(port=args.wrist_camera_port, host=args.hostname),
            "base": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
    }
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict=camera_clients, pass_sleep=True)


    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host="0.0.0.0",
        port=8000,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    iqrl_env =IQRLEnvironment(robot_env=env,args=args)
    runtime = _runtime.Runtime(
        environment=iqrl_env,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=30,
            )
        ),
        subscribers=[],
        max_hz=30,
        num_episodes=1,
        max_episode_steps=420,
    )

   

    try:
        runtime.run()
    except KeyboardInterrupt:
        print("Ctrl+C detected. Exiting...")
        iqrl_env.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
