import time
from typing import Tuple

import viser


class ViserServer(viser.ViserServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
                         
        self.gui.configure_theme(dark_mode=True)

        self.scene.add_frame("origin")
        self.scene.set_up_direction((0., 1., 0.))

    def wait_for_camera(self) -> Tuple[viser.ClientHandle, viser.CameraHandle]:
        while not list(self.get_clients().values()):
            time.sleep(0.5)

        client = list(self.get_clients().values())[-1]
        camera = client.camera

        # Used to "reset" the camera to a known state
        # camera.position = (0., 0., 0.)
        # # Point camera "down" x-axis
        # # camera.wxyz = (1., 0., 0, 0.)

        return client, camera
