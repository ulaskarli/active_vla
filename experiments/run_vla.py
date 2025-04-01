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
        args
    ) -> None:
        self._env = robot_env
        self.args=args
        self._ts = None
        self.home = [-0.035, -0.73, 0.042, 0.93, 0.017, 0.915, 0.0, 0.0]

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
        self._ts = self._env.step(action["actions"])


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
