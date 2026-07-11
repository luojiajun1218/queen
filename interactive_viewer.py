#
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

import json
import math
import os
import socket
import sys
import time
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

from arguments import ModelParams, PipelineParams, QuantizeParams
from gaussian_renderer import GaussianModel, render
from scene.cameras import MiniCam
from scene.decoders import DecoderIdentity
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov, getProjectionMatrix

pygame = None


def ensure_pygame():
    global pygame
    if pygame is None:
        import pygame as pygame_module

        pygame = pygame_module
    return pygame


def _rotation_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _numeric_key(path):
    frame_dir = path.parent.parent if path.parent.name == "compressed" else path.parent
    try:
        return int(frame_dir.name)
    except ValueError:
        return frame_dir.name


def find_frames_dir(model_path, explicit_frames_dir=None, compressed=False):
    if explicit_frames_dir:
        frames_dir = Path(explicit_frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")
        return frames_dir

    model_path = Path(model_path)
    candidates = [model_path / "frames"]
    candidates.extend(sorted(model_path.glob("*/frames")))
    pattern = "*/compressed/point_cloud.pkl" if compressed else "*/point_cloud.ply"
    for candidate in candidates:
        if any(candidate.glob(pattern)):
            return candidate
    raise FileNotFoundError(
        f"Could not find frames/{pattern} under {model_path}. "
        "Pass --frames_dir explicitly if the model uses a custom layout."
    )


def find_frame_paths(frames_dir, compressed=False):
    pattern = "*/compressed/point_cloud.pkl" if compressed else "*/point_cloud.ply"
    return sorted(Path(frames_dir).glob(pattern), key=_numeric_key)


def remember_previous_attributes(gaussians):
    for att_name in gaussians.get_atts:
        gaussians.prev_atts[att_name] = gaussians.get_decoded_atts[att_name].detach().clone()
        gaussians.prev_latents[att_name] = gaussians.get_atts[att_name].detach().clone()


def initialize_gate_flags(gaussians):
    gaussians.gate_params = OrderedDict(
        (param, gaussians.latent_args.gate_params[i] != "none")
        for i, param in enumerate(gaussians.param_names)
    )


def tensor_cache_path(cache_dir, frame_path):
    frame_id = frame_path.parent.name
    return Path(cache_dir) / f"{frame_id}.pt"


def snapshot_gaussians(gaussians):
    return {
        "active_sh_degree": gaussians.active_sh_degree,
        "latents": {
            name: tensor.detach().cpu()
            for name, tensor in gaussians.get_atts.items()
        },
    }


def restore_gaussians_from_snapshot(gaussians, snapshot):
    for name, tensor in snapshot["latents"].items():
        gaussians._latents[name] = nn.Parameter(tensor.cuda().requires_grad_(True))
        gaussians.latent_decoders[name] = DecoderIdentity()
    gaussians.gate_atts = None
    gaussians.active_sh_degree = snapshot["active_sh_degree"]
    gaussians.max_radii2D = torch.zeros((gaussians.get_xyz.shape[0]), device="cuda")
    gaussians.mask_xyz.data = torch.ones_like(gaussians._opacity).bool()
    gaussians.mask_features_dc.data = torch.ones_like(gaussians._opacity).bool()
    gaussians.mask_features_rest.data = torch.ones_like(gaussians._opacity).bool()
    gaussians.mask_scaling.data = torch.ones_like(gaussians._opacity).bool()
    gaussians.mask_rotation.data = torch.ones_like(gaussians._opacity).bool()
    gaussians.mask_opacity.data = torch.ones_like(gaussians._opacity).bool()
    gaussians.mask_flow.data = torch.ones_like(gaussians._opacity).bool()


def find_cameras_json(model_path, frames_dir, explicit_cameras=None):
    if explicit_cameras:
        cameras_path = Path(explicit_cameras)
        if not cameras_path.exists():
            raise FileNotFoundError(f"Camera file does not exist: {cameras_path}")
        return cameras_path

    candidates = [
        Path(model_path) / "cameras.json",
        frames_dir.parent / "cameras.json",
        frames_dir.parent.parent / "cameras.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find cameras.json. Pass --cameras_json explicitly.")


def load_camera_state(cameras_path, camera_id, width, height):
    with open(cameras_path, "r", encoding="utf-8") as file:
        cameras = json.load(file)
    if not cameras:
        raise RuntimeError(f"No cameras found in {cameras_path}")
    camera = cameras[min(camera_id, len(cameras) - 1)]

    src_width = float(camera["width"])
    src_height = float(camera["height"])
    fovx = focal2fov(float(camera["fx"]), src_width)
    fovy = focal2fov(float(camera["fy"]), src_height)
    if width and not height:
        height = max(1, round(width * src_height / src_width))
    elif height and not width:
        width = max(1, round(height * src_width / src_height))
    elif not width and not height:
        width, height = int(src_width), int(src_height)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = np.asarray(camera["rotation"], dtype=np.float32)
    c2w[:3, 3] = np.asarray(camera["position"], dtype=np.float32)
    return c2w, fovx, fovy, int(width), int(height)


class GaussianFrameCache:
    def __init__(self, frame_paths, lp_args, qp_args, cache_size, tensor_cache_dir=None, build_tensor_cache=False):
        self.frame_paths = frame_paths
        self.lp_args = lp_args
        self.qp_args = qp_args
        self.cache_size = max(1, cache_size)
        self.tensor_cache_dir = Path(tensor_cache_dir) if tensor_cache_dir else None
        self.build_tensor_cache = build_tensor_cache
        if self.tensor_cache_dir:
            self.tensor_cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = OrderedDict()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures = {}

    def close(self):
        self.executor.shutdown(wait=False, cancel_futures=True)

    def contains(self, index):
        return index % len(self.frame_paths) in self.cache

    def _load_uncached(self, index):
        start = time.perf_counter()
        gaussians = GaussianModel(self.lp_args.sh_degree, self.qp_args, self.lp_args)
        cache_path = tensor_cache_path(self.tensor_cache_dir, self.frame_paths[index]) if self.tensor_cache_dir else None
        if cache_path and cache_path.exists():
            snapshot = torch.load(cache_path, map_location="cpu", weights_only=False)
            restore_gaussians_from_snapshot(gaussians, snapshot)
        else:
            gaussians.load_ply(str(self.frame_paths[index]))
            if cache_path and self.build_tensor_cache:
                torch.save(snapshot_gaussians(gaussians), cache_path)
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - start) * 1000.0
        return index, gaussians, load_ms

    def _insert(self, index, gaussians):
        self.cache[index] = gaussians
        self.cache.move_to_end(index)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def prefetch(self, indices):
        for index in indices:
            index %= len(self.frame_paths)
            if index in self.cache or index in self.futures:
                continue
            self.futures[index] = self.executor.submit(self._load_uncached, index)

    def collect_ready(self):
        ready = []
        for index, future in list(self.futures.items()):
            if not future.done():
                continue
            del self.futures[index]
            loaded_index, gaussians, load_ms = future.result()
            self._insert(loaded_index, gaussians)
            ready.append((loaded_index, load_ms))
        return ready

    def get(self, index):
        index %= len(self.frame_paths)
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index], 0.0, True
        if index in self.futures:
            loaded_index, gaussians, load_ms = self.futures.pop(index).result()
        else:
            loaded_index, gaussians, load_ms = self._load_uncached(index)
        self._insert(loaded_index, gaussians)
        return gaussians, load_ms, False

    def profile_load(self, index):
        index %= len(self.frame_paths)
        timings = {}
        start_total = time.perf_counter()

        start = time.perf_counter()
        gaussians = GaussianModel(self.lp_args.sh_degree, self.qp_args, self.lp_args)
        timings["gaussian_model_init_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        cache_path = tensor_cache_path(self.tensor_cache_dir, self.frame_paths[index]) if self.tensor_cache_dir else None
        if cache_path and cache_path.exists():
            snapshot = torch.load(cache_path, map_location="cpu", weights_only=False)
            restore_gaussians_from_snapshot(gaussians, snapshot)
            timings["load_tensor_cache_submit_ms"] = (time.perf_counter() - start) * 1000.0
        else:
            gaussians.load_ply(str(self.frame_paths[index]))
            if cache_path and self.build_tensor_cache:
                torch.save(snapshot_gaussians(gaussians), cache_path)
            timings["load_ply_submit_ms"] = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        torch.cuda.synchronize()
        timings["load_cuda_sync_ms"] = (time.perf_counter() - start) * 1000.0
        timings["load_total_ms"] = (time.perf_counter() - start_total) * 1000.0
        return gaussians, timings


class ViewerState:
    def __init__(self, c2w, fovx, fovy, width, height, znear, zfar):
        self.initial_c2w = c2w.copy()
        self.c2w = c2w.copy()
        self.fovx = fovx
        self.fovy = fovy
        self.width = width
        self.height = height
        self.znear = znear
        self.zfar = zfar
        self.frame_index = 0
        self.playing = False
        self.dirty = True
        self.lock_mouse = False

    def reset_camera(self):
        self.c2w = self.initial_c2w.copy()
        self.dirty = True

    def make_camera(self):
        w2c = np.linalg.inv(self.c2w).astype(np.float32)
        world_view = torch.tensor(w2c, device="cuda").transpose(0, 1)
        proj = getProjectionMatrix(self.znear, self.zfar, self.fovx, self.fovy).transpose(0, 1).cuda()
        full_proj = world_view.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        center = torch.tensor(self.c2w[:3, 3], dtype=torch.float32, device="cuda")
        return MiniCam(0, self.width, self.height, self.fovy, self.fovx, self.znear, self.zfar, world_view, full_proj, center, self.frame_index)

    def rotate(self, dx, dy, sensitivity, invert_x=False, invert_y=False):
        sx = -1.0 if invert_x else 1.0
        sy = 1.0 if invert_y else -1.0
        yaw = _rotation_y(dx * sensitivity * sx)
        pitch = _rotation_x(dy * sensitivity * sy)
        self.c2w[:3, :3] = yaw @ self.c2w[:3, :3] @ pitch
        self.dirty = True

    def move(self, local_delta):
        self.c2w[:3, 3] += self.c2w[:3, :3] @ local_delta
        self.dirty = True


def tensor_to_surface(image):
    pygame = ensure_pygame()
    image = image.detach().clamp(0.0, 1.0).mul(255).byte()
    array = image.permute(1, 2, 0).contiguous().cpu().numpy()
    return pygame.image.frombuffer(array.tobytes(), (array.shape[1], array.shape[0]), "RGB")


def handle_input(state, args, dt):
    pygame = ensure_pygame()
    keys = pygame.key.get_pressed()
    speed = args.move_speed * dt
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        speed *= args.fast_multiplier

    delta = np.zeros(3, dtype=np.float32)
    if keys[pygame.K_w]:
        delta[2] += speed
    if keys[pygame.K_s]:
        delta[2] -= speed
    if keys[pygame.K_d]:
        delta[0] += speed
    if keys[pygame.K_a]:
        delta[0] -= speed
    if keys[pygame.K_e]:
        delta[1] += speed
    if keys[pygame.K_q] or keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
        delta[1] -= speed
    if np.any(delta):
        state.move(delta)


def set_mouse_lock(state, enabled):
    pygame = ensure_pygame()
    state.lock_mouse = enabled
    pygame.event.set_grab(enabled)
    pygame.mouse.set_visible(not enabled)
    pygame.mouse.get_rel()


def mean_ms(values):
    return float(np.mean(values)) if values else 0.0


def update_title(args, state, total_frames, render_ms, load_ms, cache_hit, loop_fps=None, surface_ms=None, flip_ms=None):
    pygame = ensure_pygame()
    mode = "play" if state.playing else "pause"
    cached = "cache" if cache_hit else "load"
    fps = 1000.0 / render_ms if render_ms > 0 else 0.0
    pygame.display.set_caption(
        f"QUEEN interactive viewer | frame {state.frame_index + 1}/{total_frames} | "
        f"{mode} | render {render_ms:.1f} ms ({fps:.1f} FPS) | {cached} {load_ms:.1f} ms | "
        f"{args.width}x{args.height or 'auto'}"
    )


def benchmark_compressed_sequence(args, lp_args, pp_args, qp_args, frames_dir, state, background):
    init_ply = Path(frames_dir) / "0001" / "point_cloud.ply"
    compressed_paths = find_frame_paths(frames_dir, compressed=True)
    if not init_ply.exists():
        raise FileNotFoundError(f"Compressed benchmark needs the first dense frame: {init_ply}")
    if not compressed_paths:
        raise FileNotFoundError(f"No compressed frames found under {frames_dir}")

    gaussians = GaussianModel(lp_args.sh_degree, qp_args, lp_args)
    initialize_gate_flags(gaussians)
    with torch.inference_mode():
        start = time.perf_counter()
        gaussians.load_ply(str(init_ply))
        initialize_gate_flags(gaussians)
        torch.cuda.synchronize()
        init_ms = (time.perf_counter() - start) * 1000.0
        remember_previous_attributes(gaussians)

        load_times = []
        render_times = []
        total_frames = min(args.compressed_benchmark_frames, len(compressed_paths))
        for path in compressed_paths[:total_frames]:
            gaussians.frame_idx = _numeric_key(path)
            start = time.perf_counter()
            gaussians.load_compressed_pkl(str(path))
            initialize_gate_flags(gaussians)
            torch.cuda.synchronize()
            load_times.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            render(state.make_camera(), gaussians, pp_args, background)["render"]
            torch.cuda.synchronize()
            render_times.append((time.perf_counter() - start) * 1000.0)
            remember_previous_attributes(gaussians)

    load_ms = float(np.mean(load_times))
    render_ms = float(np.mean(render_times))
    total_ms = load_ms + render_ms
    print(f"Compressed sequential benchmark frames={total_frames} resolution={state.width}x{state.height}")
    print(f"init_frame1_ply_ms={init_ms:.2f}")
    print(f"compressed_load_mean_ms={load_ms:.2f}")
    print(f"compressed_load_p50_ms={float(np.median(load_times)):.2f}")
    print(f"compressed_render_mean_ms={render_ms:.2f}")
    print(f"compressed_total_mean_ms={total_ms:.2f}")
    print(f"compressed_total_fps={1000.0 / total_ms:.2f}")


def run_viewer(args, lp_args, pp_args, qp_args):
    qp_args.seed = lp_args.seed
    frames_dir = find_frames_dir(args.model_path, args.frames_dir, compressed=args.compressed)
    frame_paths = find_frame_paths(frames_dir, compressed=False)
    if not frame_paths:
        raise RuntimeError(f"No point_cloud.ply files found in {frames_dir}")

    cameras_path = find_cameras_json(args.model_path, frames_dir, args.cameras_json)
    c2w, fovx, fovy, width, height = load_camera_state(cameras_path, args.camera_id, args.width, args.height)
    state = ViewerState(c2w, fovx, fovy, width, height, lp_args.znear, lp_args.zfar)

    cache = GaussianFrameCache(
        frame_paths,
        lp_args,
        qp_args,
        args.cache_frames,
        tensor_cache_dir=args.tensor_cache_dir,
        build_tensor_cache=args.build_tensor_cache,
    )
    background = torch.tensor([1, 1, 1] if lp_args.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    if args.compressed_benchmark_frames > 0:
        benchmark_compressed_sequence(args, lp_args, pp_args, qp_args, frames_dir, state, background)
        return

    if args.smoke_test:
        with torch.inference_mode():
            gaussians, load_ms, _ = cache.get(0)
            start = time.perf_counter()
            image = render(state.make_camera(), gaussians, pp_args, background)["render"]
            torch.cuda.synchronize()
            render_ms = (time.perf_counter() - start) * 1000.0
        print(f"Smoke test OK: frame=1 size={tuple(image.shape)} load_ms={load_ms:.1f} render_ms={render_ms:.1f}")
        return

    if args.benchmark_iters > 0:
        with torch.inference_mode():
            gaussians, load_timings = cache.profile_load(0)

            camera_times = []
            for _ in range(args.benchmark_iters):
                start = time.perf_counter()
                camera = state.make_camera()
                torch.cuda.synchronize()
                camera_times.append((time.perf_counter() - start) * 1000.0)

            camera = state.make_camera()
            for _ in range(args.benchmark_warmup):
                image = render(camera, gaussians, pp_args, background)["render"]
            torch.cuda.synchronize()

            render_submit_times = []
            render_sync_times = []
            render_total_times = []
            for _ in range(args.benchmark_iters):
                start_total = time.perf_counter()
                start = time.perf_counter()
                image = render(camera, gaussians, pp_args, background)["render"]
                render_submit_times.append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                torch.cuda.synchronize()
                render_sync_times.append((time.perf_counter() - start) * 1000.0)
                render_total_times.append((time.perf_counter() - start_total) * 1000.0)

            pygame = ensure_pygame()
            pygame.init()
            transfer_times = []
            surface_times = []
            for _ in range(args.benchmark_iters):
                start = time.perf_counter()
                image_cpu = image.detach().clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
                transfer_times.append((time.perf_counter() - start) * 1000.0)
                start = time.perf_counter()
                pygame.surfarray.make_surface(np.swapaxes(image_cpu, 0, 1))
                surface_times.append((time.perf_counter() - start) * 1000.0)
            pygame.quit()

        camera_ms = float(np.mean(camera_times))
        render_submit_ms = float(np.mean(render_submit_times))
        render_sync_ms = float(np.mean(render_sync_times))
        render_ms = float(np.mean(render_total_times))
        render_p50 = float(np.median(render_total_times))
        transfer_ms = float(np.mean(transfer_times))
        surface_ms = float(np.mean(surface_times))
        total_ms = camera_ms + render_ms + transfer_ms + surface_ms
        print(f"Benchmark frame=1 resolution={width}x{height}")
        for key, value in load_timings.items():
            print(f"{key}={value:.2f}")
        print(f"camera_build_mean_ms={camera_ms:.2f}")
        print(f"render_submit_mean_ms={render_submit_ms:.2f}")
        print(f"render_cuda_sync_mean_ms={render_sync_ms:.2f}")
        print(f"render_total_mean_ms={render_ms:.2f}")
        print(f"render_total_p50_ms={render_p50:.2f}")
        print(f"render_only_fps={1000.0 / render_ms:.2f}")
        print(f"gpu_to_cpu_numpy_mean_ms={transfer_ms:.2f}")
        print(f"pygame_surface_mean_ms={surface_ms:.2f}")
        print(f"viewer_cached_frame_total_mean_ms={total_ms:.2f}")
        print(f"viewer_cached_frame_fps={1000.0 / total_ms:.2f}")
        print(f"playback_with_ply_load_fps_upper_bound={1000.0 / (load_timings['load_total_ms'] + total_ms):.2f}")
        return

    preload_count = min(args.preload_frames, len(frame_paths), args.cache_frames)
    if preload_count > 0:
        print(f"Preloading {preload_count} frame(s) into the GPU cache...", flush=True)
        for index in range(preload_count):
            _, load_ms, cache_hit = cache.get(index)
            if not cache_hit:
                print(f"  frame {index + 1}/{preload_count}: {load_ms:.1f} ms", flush=True)
    cache.prefetch(range(0, min(args.prefetch_ahead + 1, len(frame_paths))))

    pygame = ensure_pygame()
    pygame.init()
    flags = pygame.HWSURFACE | pygame.DOUBLEBUF
    screen = pygame.display.set_mode((width, height), flags)
    clock = pygame.time.Clock()
    pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION])

    last_advance = time.perf_counter()
    surface = None
    render_ms = 0.0
    load_ms = 0.0
    cache_hit = False
    last_surface_ms = 0.0
    last_flip_ms = 0.0
    running = True
    stat_keys = ["loop", "collect", "input", "prefetch", "load", "camera", "render", "surface", "flip"]
    stats = {key: deque(maxlen=args.stats_samples) for key in stat_keys}
    last_stats_print = time.perf_counter()

    print("Controls: WASD move, mouse drag look, E up, Q/Ctrl down, Shift fast", flush=True)
    print("Playback: Space/P pause/play, Left/Right step frame, R reset camera, M lock mouse, Esc quit", flush=True)
    print(f"Frames: {frames_dir}", flush=True)
    print(f"Cameras: {cameras_path}", flush=True)

    with torch.inference_mode():
        while running:
            loop_start = time.perf_counter()
            dt = clock.tick(args.max_display_fps) / 1000.0
            now = time.perf_counter()
            start = time.perf_counter()
            cache.collect_ready()
            stats["collect"].append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in (pygame.K_p, pygame.K_SPACE):
                        state.playing = not state.playing
                    elif event.key == pygame.K_RIGHT:
                        state.frame_index = (state.frame_index + 1) % len(frame_paths)
                        state.dirty = True
                    elif event.key == pygame.K_LEFT:
                        state.frame_index = (state.frame_index - 1) % len(frame_paths)
                        state.dirty = True
                    elif event.key == pygame.K_r:
                        state.reset_camera()
                    elif event.key == pygame.K_m:
                        set_mouse_lock(state, not state.lock_mouse)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                    set_mouse_lock(state, not state.lock_mouse)
                elif event.type == pygame.MOUSEMOTION and state.lock_mouse:
                    dx, dy = event.rel
                    state.rotate(dx, dy, args.mouse_sensitivity, args.invert_mouse_x, args.invert_mouse_y)

            handle_input(state, args, dt)
            stats["input"].append((time.perf_counter() - start) * 1000.0)

            if state.playing and now - last_advance >= 1.0 / max(1.0, args.play_fps):
                next_frame = (state.frame_index + 1) % len(frame_paths)
                if cache.contains(next_frame):
                    state.frame_index = next_frame
                    state.dirty = True
                    last_advance = now
                else:
                    cache.prefetch([next_frame])
                    last_advance = now

            start = time.perf_counter()
            prefetch_indices = [
                (state.frame_index + offset) % len(frame_paths)
                for offset in range(1, args.prefetch_ahead + 1)
            ]
            cache.prefetch(prefetch_indices)
            stats["prefetch"].append((time.perf_counter() - start) * 1000.0)

            if state.dirty or surface is None or args.continuous_render:
                start = time.perf_counter()
                gaussians, load_ms, cache_hit = cache.get(state.frame_index)
                stats["load"].append((time.perf_counter() - start) * 1000.0)

                start = time.perf_counter()
                camera = state.make_camera()
                torch.cuda.synchronize()
                stats["camera"].append((time.perf_counter() - start) * 1000.0)

                start = time.perf_counter()
                image = render(camera, gaussians, pp_args, background)["render"]
                torch.cuda.synchronize()
                render_ms = (time.perf_counter() - start) * 1000.0
                stats["render"].append(render_ms)

                start = time.perf_counter()
                surface = tensor_to_surface(image)
                last_surface_ms = (time.perf_counter() - start) * 1000.0
                stats["surface"].append(last_surface_ms)
                state.dirty = False

            start = time.perf_counter()
            screen.blit(surface, (0, 0))
            pygame.display.flip()
            last_flip_ms = (time.perf_counter() - start) * 1000.0
            stats["flip"].append(last_flip_ms)
            stats["loop"].append((time.perf_counter() - loop_start) * 1000.0)

            loop_ms = mean_ms(stats["loop"])
            loop_fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
            update_title(args, state, len(frame_paths), render_ms, load_ms, cache_hit, loop_fps, last_surface_ms, last_flip_ms)

            if args.stats_interval > 0 and time.perf_counter() - last_stats_print >= args.stats_interval:
                print(
                    "stats "
                    f"loop={loop_ms:.2f}ms/{loop_fps:.1f}fps "
                    f"load={mean_ms(stats['load']):.2f} "
                    f"camera={mean_ms(stats['camera']):.2f} "
                    f"render={mean_ms(stats['render']):.2f} "
                    f"surface={mean_ms(stats['surface']):.2f} "
                    f"flip={mean_ms(stats['flip']):.2f} "
                    f"collect={mean_ms(stats['collect']):.2f} "
                    f"input={mean_ms(stats['input']):.2f} "
                    f"prefetch={mean_ms(stats['prefetch']):.2f} "
                    f"frame={state.frame_index + 1}/{len(frame_paths)} "
                    f"cache={'hit' if cache_hit else 'miss'}",
                    flush=True,
                )
                last_stats_print = time.perf_counter()

    cache.close()
    pygame.quit()


def parse_args():
    print("Running on ", socket.gethostname(), flush=True)
    default_config = "configs/dynerf.yaml"
    config_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else default_config
    if config_path:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        config = {}
    config = defaultdict(lambda: {}, config)

    parser = ArgumentParser(description="Interactive QUEEN Gaussian viewer")
    lp = ModelParams(parser, config["model_params"])
    pp = PipelineParams(parser, config["pipe_params"])
    qp = QuantizeParams(parser, config["quantize_params"])

    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--frames_dir", type=str, default=None)
    parser.add_argument("--cameras_json", type=str, default=None)
    parser.add_argument("--camera_id", type=int, default=0)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--play_fps", type=float, default=24.0)
    parser.add_argument("--max_display_fps", type=int, default=120)
    parser.add_argument("--cache_frames", type=int, default=96)
    parser.add_argument("--prefetch_ahead", type=int, default=24)
    parser.add_argument("--preload_frames", type=int, default=96)
    parser.add_argument("--tensor_cache_dir", type=str, default=None)
    parser.add_argument("--build_tensor_cache", action="store_true")
    parser.add_argument("--move_speed", type=float, default=1.5)
    parser.add_argument("--fast_multiplier", type=float, default=4.0)
    parser.add_argument("--mouse_sensitivity", type=float, default=0.002)
    parser.add_argument("--invert_mouse_x", action="store_true")
    parser.add_argument("--invert_mouse_y", action="store_true")
    parser.add_argument("--continuous_render", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--compressed_benchmark_frames", type=int, default=0)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--benchmark_iters", type=int, default=0)
    parser.add_argument("--benchmark_warmup", type=int, default=10)
    parser.add_argument("--stats_interval", type=float, default=1.0)
    parser.add_argument("--stats_samples", type=int, default=120)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if not args.source_path:
        args.source_path = "data/dynerf/coffee_martini"
    if not args.model_path:
        args.model_path = "./output/coffee_martini_trained/queen_compressed_coffee_martini"
    if args.tensor_cache_dir is None:
        args.tensor_cache_dir = os.path.join(args.model_path, "tensor_cache")
    return args, lp.extract(args), pp.extract(args), qp.extract(args)


if __name__ == "__main__":
    viewer_args, model_args, pipe_args, quant_args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("interactive_viewer.py requires a CUDA GPU because it uses QUEEN's CUDA Gaussian renderer.")
    safe_state(viewer_args.quiet)
    run_viewer(viewer_args, model_args, pipe_args, quant_args)
