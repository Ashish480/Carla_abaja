#!/usr/bin/env python

from __future__ import print_function
import argparse
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import numpy as np

try:
    import pygame
    from pygame.locals import K_ESCAPE, K_q, KMOD_CTRL
except ImportError:
    raise RuntimeError('Cannot import pygame, ensure it is installed.')

# CARLA imports
try:
    sys.path.append(
        glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
            sys.version_info.major,
            sys.version_info.minor,
            'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from ultralytics import YOLO  # YOLOv8 integration


# ==============================================================================
# -- Helper Functions ----------------------------------------------------------
# ==============================================================================

def find_weather_presets():
    """Retrieve weather presets."""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Retrieve the display name of an actor."""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + '…') if len(name) > truncate else name


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD:
    """Class for HUD text."""

    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.notifications = FadingText(font, (width, 40), (0, height - 40))
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0

    def on_world_tick(self, timestamp):
        """Callback for world tick."""
        self.server_fps = 1.0 / timestamp.delta_seconds
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def notification(self, text, seconds=2.0):
        """Show notifications."""
        self.notifications.set_text(text, seconds=seconds)

    def render(self, display):
        """Render the HUD."""
        self.notifications.render(display)


class FadingText:
    """Class for fading text notifications."""

    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, seconds=2.0):
        """Set fading text."""
        text_texture = self.font.render(text, True, (255, 255, 255))
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))
        self.seconds_left = seconds

    def render(self, display):
        """Render fading text."""
        if self.seconds_left > 0:
            display.blit(self.surface, self.pos)
            self.seconds_left -= 0.02


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================

class CameraManager:
    """Class for managing camera sensors."""

    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self.hud = hud
        self.parent = parent_actor
        self.recording = False
        self.yolo_model = YOLO("yolov8n.pt")  # Load YOLOv8 model

    def set_sensor(self, index):
        """Set the current sensor."""
        # Add logic for sensor configuration if needed
        pass

    @staticmethod
    def _parse_image(weak_self, image):
        """Process the image data."""
        self = weak_self()
        if not self:
            return

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))
        rgb_image = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB

        # YOLOv8 Object Detection
        results = self.yolo_model.predict(source=rgb_image, conf=0.5)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = box.conf[0]
                color = (255, 0, 0)
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    rgb_image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

        # Render to pygame surface
        self.surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0, 1))


# ==============================================================================
# -- Main Loop -----------------------------------------------------------------
# ==============================================================================

def game_loop(args):
    """Main loop for the simulation."""
    pygame.init()
    pygame.font.init()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        hud = HUD(args.width, args.height)
        display = pygame.display.set_mode((args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        camera_manager = CameraManager(world, hud)

        while True:
            world.tick()
            hud.on_world_tick(world.wait_for_tick())
            display.fill((0, 0, 0))
            hud.render(display)
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

    finally:
        pygame.quit()


def main():
    """Main entry point."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('--res', default='1280x720', help='Window resolution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()

