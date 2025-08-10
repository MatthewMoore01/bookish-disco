"""
Radon transform and filtered back-projection demo.

This script provides a small desktop application illustrating how
computed tomography collects one-dimensional projections of an
attenuation map (Radon transform) and reconstructs it using
filtered back-projection (inverse Radon).

How to run
----------
$ python demo_radon.py

The interface uses matplotlib for visualisation and widgets.  It
relies only on numpy, matplotlib, scikit-image and Pillow for image
I/O.  No web frameworks are required.
"""

import math
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import (
    Slider,
    Button,
    RadioButtons,
    CheckButtons,
    TextBox,
)
from skimage.transform import radon, iradon, resize
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def compute_projection(img: np.ndarray, theta_deg: float, n_detectors: int) -> np.ndarray:
    """Compute line integral for a single angle using the Radon transform.

    Parameters
    ----------
    img : ndarray
        Source image (float32, scaled to [0,1]).
    theta_deg : float
        Angle in degrees.
    n_detectors : int
        Number of detector samples to resample the projection to.
    """
    theta = np.array([theta_deg])
    # radon returns (max(image.shape), len(theta))
    L = radon(img, theta=theta, circle=True, preserve_range=True)
    L = L[:, 0]
    if n_detectors != len(L):
        x = np.linspace(0, len(L) - 1, len(L))
        x_new = np.linspace(0, len(L) - 1, n_detectors)
        L = np.interp(x_new, x, L)
    return L.astype(np.float32)


def append_measurement(
    sino: np.ndarray | None, theta_list: list[float], L: np.ndarray, theta: float
):
    """Append or replace a measurement in the sinogram."""
    theta = float(theta)
    if sino is None:
        sino = L[:, np.newaxis]
        theta_list[:] = [theta]
        return sino
    angles = np.array(theta_list)
    idx = np.where(np.isclose(angles, theta))[0]
    if len(idx):
        sino[:, idx[0]] = L
    else:
        sino = np.column_stack([sino, L])
        theta_list.append(theta)
    return sino


def reconstruct(
    sino: np.ndarray,
    theta_list: list[float],
    filter_name: str,
    interpolation: str,
    output_size: int,
    apply_mask: bool = True,
) -> np.ndarray:
    """Run filtered back-projection using scikit-image's iradon."""
    if sino is None or len(theta_list) < 2:
        return None
    angles = np.array(theta_list)
    order = np.argsort(angles)
    sino = sino[:, order]
    angles = angles[order]
    filt = None if filter_name == "none" else filter_name
    recon = iradon(
        sino,
        theta=angles,
        circle=apply_mask,
        filter_name=filt,
        interpolation=interpolation,
        output_size=output_size,
        preserve_range=True,
    )
    return recon.astype(np.float32)


def load_image(path: str, size: int) -> np.ndarray:
    """Load an external image and convert to grayscale [0,1]."""
    img = Image.open(path).convert("L")
    img = np.array(img, dtype=np.float32) / 255.0
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = resize(img, (size, size), mode="reflect", anti_aliasing=True).astype(
        np.float32
    )
    return img


def save_npz(path: str, sino: np.ndarray, theta_deg: list[float]):
    np.savez(path, sino=np.asarray(sino, dtype=np.float32), theta_deg=np.asarray(theta_deg, dtype=np.float32))


def load_npz(path: str):
    data = np.load(path)
    sino = data["sino"].astype(np.float32)
    theta = data["theta_deg"].astype(np.float32).tolist()
    return sino, theta


def export_pngs(img, sino, recon):
    plt.imsave("source.png", img, cmap="gray")
    if sino is not None:
        plt.imsave("sinogram.png", sino, cmap="gray")
    if recon is not None:
        plt.imsave("reconstruction.png", recon, cmap="gray")


# -----------------------------------------------------------------------------
# Application class
# -----------------------------------------------------------------------------


@dataclass
class RadonState:
    img: np.ndarray
    theta_rec: list
    sino: np.ndarray | None
    recon: np.ndarray | None
    n_detectors: int
    recon_size: int
    sweep_step: float
    show_intensity: bool
    noise_level: str
    filter_name: str
    interpolation: str
    mask: bool = True


class RadonDemo:
    def __init__(self):
        # initial image: Shepp-Logan phantom
        phantom = data.shepp_logan_phantom().astype(np.float32)
        phantom = resize(phantom, (256, 256), mode="reflect", anti_aliasing=True)
        self.state = RadonState(
            img=phantom,
            theta_rec=[],
            sino=None,
            recon=None,
            n_detectors=384,
            recon_size=256,
            sweep_step=1.0,
            show_intensity=False,
            noise_level="none",
            filter_name="ramp",
            interpolation="linear",
            mask=True,
        )
        self.current_angle = 0.0
        self._build_ui()
        self.update_projection()

    # ------------------------------------------------------------------
    def _build_ui(self):
        fig = plt.figure("Radon demo", figsize=(12, 6))
        self.fig = fig
        gs = fig.add_gridspec(2, 3, height_ratios=[8, 2])
        self.ax_sino = fig.add_subplot(gs[0, 0])
        self.ax_img = fig.add_subplot(gs[0, 1])
        self.ax_proj = fig.add_subplot(gs[0, 2])
        self.ax_sino.set_title("Sinogram")
        self.ax_img.set_title("Source / Reconstruction")
        self.ax_proj.set_title("Projection at θ = 0°")

        # Display initial images
        self.img_im = self.ax_img.imshow(self.state.img, cmap="gray", vmin=0, vmax=1)
        self.ax_img.axis("off")
        self.sino_im = self.ax_sino.imshow(
            np.zeros((self.state.n_detectors, 1)),
            cmap="gray",
            aspect="auto",
            vmin=0,
            vmax=1,
        )
        self.ax_sino.set_xlabel("Angle index")
        self.ax_sino.set_ylabel("Detector")
        self.proj_line, = self.ax_proj.plot([], [])
        self.ax_proj.set_xlim(0, self.state.n_detectors)
        self.ax_proj.set_ylim(-1, 1)
        self.ax_proj.set_ylabel("Line integral")

        # Geometry overlay
        self.detector_line = self.ax_img.plot([], [], color="r")[0]
        self.ray_lines = [self.ax_img.plot([], [], color="y", lw=0.5)[0] for _ in range(3)]

        # Controls
        ax_slider = fig.add_subplot(gs[1, 0:3])
        ax_slider.axis("off")

        slider_ax = fig.add_axes([0.15, 0.1, 0.5, 0.03])
        self.slider = Slider(slider_ax, "Angle [deg]", 0.0, 180.0, valinit=0.0, valstep=0.5)
        self.slider.on_changed(self.on_angle_change)

        btn_ax = fig.add_axes([0.68, 0.095, 0.08, 0.04])
        self.btn_play = Button(btn_ax, "Play")
        self.btn_play.on_clicked(self.toggle_play)

        btn_rec_ax = fig.add_axes([0.77, 0.095, 0.09, 0.04])
        self.btn_record = Button(btn_rec_ax, "Record current")
        self.btn_record.on_clicked(self.record_current)

        btn_sweep_ax = fig.add_axes([0.87, 0.095, 0.08, 0.04])
        self.btn_sweep = Button(btn_sweep_ax, "Record sweep")
        self.btn_sweep.on_clicked(self.record_sweep)

        # second row of buttons and options
        y2 = 0.02
        x = 0.01
        self.btn_clear = Button(fig.add_axes([x, y2, 0.08, 0.04]), "Clear sinogram")
        self.btn_clear.on_clicked(self.clear_sinogram)
        x += 0.09
        self.btn_recon = Button(fig.add_axes([x, y2, 0.08, 0.04]), "Reconstruct")
        self.btn_recon.on_clicked(self.do_reconstruct)
        x += 0.09
        self.btn_load_img = Button(fig.add_axes([x, y2, 0.08, 0.04]), "Load image")
        self.btn_load_img.on_clicked(self.load_image_dialog)
        x += 0.09
        self.btn_save_sino = Button(fig.add_axes([x, y2, 0.08, 0.04]), "Save sinogram")
        self.btn_save_sino.on_clicked(self.save_sinogram)
        x += 0.09
        self.btn_load_sino = Button(fig.add_axes([x, y2, 0.08, 0.04]), "Load sinogram")
        self.btn_load_sino.on_clicked(self.load_sinogram)
        x += 0.09
        self.btn_export = Button(fig.add_axes([x, y2, 0.08, 0.04]), "Export PNGs")
        self.btn_export.on_clicked(self.export_pngs)
        x += 0.09
        self.btn_reset = Button(fig.add_axes([x, y2, 0.07, 0.04]), "Reset view")
        self.btn_reset.on_clicked(self.reset_view)

        # Filter radio buttons
        self.radio_filter = RadioButtons(
            fig.add_axes([0.77, 0.01, 0.07, 0.11]),
            ("ramp", "shepp-logan", "hann", "hamming", "cosine", "none"),
            active=0,
        )
        self.radio_filter.on_clicked(self.on_filter_change)
        self.radio_filter.ax.set_title("Filter")

        # Interpolation radio buttons
        self.radio_interp = RadioButtons(
            fig.add_axes([0.85, 0.01, 0.06, 0.08]),
            ("linear", "nearest"),
            active=0,
        )
        self.radio_interp.on_clicked(self.on_interp_change)
        self.radio_interp.ax.set_title("Interp")

        # Show intensity checkbox
        self.chk_intensity = CheckButtons(
            fig.add_axes([0.92, 0.04, 0.07, 0.05]), ["Show intensity"], [False]
        )
        self.chk_intensity.on_clicked(self.on_show_intensity)

        # Noise level
        self.radio_noise = RadioButtons(
            fig.add_axes([0.92, 0.01, 0.07, 0.08]),
            ("none", "low", "medium", "high"),
            active=0,
        )
        self.radio_noise.on_clicked(self.on_noise_change)
        self.radio_noise.ax.set_title("Noise")

        # Numeric controls using TextBox
        self.tb_detectors = TextBox(fig.add_axes([0.15, 0.02, 0.05, 0.03]), "Detectors", str(self.state.n_detectors))
        self.tb_detectors.on_submit(self.on_detectors_submit)
        self.tb_recon = TextBox(fig.add_axes([0.22, 0.02, 0.05, 0.03]), "Recon", str(self.state.recon_size))
        self.tb_recon.on_submit(self.on_recon_size_submit)
        self.tb_sweep = TextBox(fig.add_axes([0.29, 0.02, 0.05, 0.03]), "Step", str(self.state.sweep_step))
        self.tb_sweep.on_submit(self.on_sweep_step_submit)

        # Image choice radio (phantom/smiley)
        self.radio_img = RadioButtons(
            fig.add_axes([0.36, 0.02, 0.07, 0.07]),
            ("phantom", "smiley"),
            active=0,
        )
        self.radio_img.on_clicked(self.on_image_choice)
        self.radio_img.ax.set_title("Image")

        # Mask toggle
        self.chk_mask = CheckButtons(fig.add_axes([0.44, 0.02, 0.06, 0.05]), ["Mask"], [True])
        self.chk_mask.on_clicked(self.on_mask_toggle)

        # Timer for play mode
        self.timer = fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.timer_update)

    # ------------------------------------------------------------------
    # UI callbacks
    def on_angle_change(self, val):
        self.current_angle = val
        self.ax_proj.set_title(f"Projection at θ = {val:.1f}°")
        self.update_projection()

    def toggle_play(self, event):
        self.playing = not getattr(self, "playing", False)
        if self.playing:
            self.btn_play.label.set_text("Pause")
            self.timer.start()
        else:
            self.btn_play.label.set_text("Play")
            self.timer.stop()

    def timer_update(self):
        val = self.slider.val + 1.0
        if val > 180:
            val = 0.0
        self.slider.set_val(val)

    def update_projection(self):
        L = compute_projection(
            self.state.img, self.current_angle, self.state.n_detectors
        )
        if self.state.show_intensity:
            I = np.exp(-L)
            sigma = {"none": 0.0, "low": 0.01, "medium": 0.03, "high": 0.05}[
                self.state.noise_level
            ]
            if sigma > 0:
                I = np.clip(I + np.random.normal(0, sigma, size=I.shape), 0, 1)
            y = I
            self.ax_proj.set_ylim(0, 1.1)
            self.ax_proj.set_ylabel("Intensity")
        else:
            y = L
            self.ax_proj.set_ylim(L.min() - 0.1, L.max() + 0.1)
            self.ax_proj.set_ylabel("Line integral")
        self.proj_line.set_data(np.arange(len(y)), y)
        self.ax_proj.set_xlim(0, len(y))
        self.fig.canvas.draw_idle()
        self.update_geometry()

    def update_geometry(self):
        h, w = self.state.img.shape
        cx, cy = w / 2, h / 2
        radius = math.hypot(w, h) / 2
        ang = math.radians(self.current_angle)
        dx = math.cos(ang)
        dy = math.sin(ang)
        x0, x1 = cx - dx * radius, cx + dx * radius
        y0, y1 = cy - dy * radius, cy + dy * radius
        self.detector_line.set_data([x0, x1], [y0, y1])
        # Rays perpendicular to detector axis
        nx, ny = -dy, dx
        offsets = np.linspace(-w / 4, w / 4, 3)
        for off, line in zip(offsets, self.ray_lines):
            sx = cx + dx * off
            sy = cy + dy * off
            rx0, rx1 = sx - nx * radius, sx + nx * radius
            ry0, ry1 = sy - ny * radius, sy + ny * radius
            line.set_data([rx0, rx1], [ry0, ry1])

    def record_current(self, event):
        L = compute_projection(
            self.state.img, self.current_angle, self.state.n_detectors
        )
        self.state.sino = append_measurement(
            self.state.sino, self.state.theta_rec, L, self.current_angle
        )
        self.redraw_sinogram()

    def record_sweep(self, event):
        step = self.state.sweep_step
        angles = np.arange(0.0, 180.0, step)
        for i, th in enumerate(angles):
            L = compute_projection(self.state.img, th, self.state.n_detectors)
            self.state.sino = append_measurement(
                self.state.sino, self.state.theta_rec, L, th
            )
            if i % 5 == 0:
                self.slider.set_val(th)
                self.redraw_sinogram()
        self.slider.set_val(angles[-1] if len(angles) else 0.0)
        self.redraw_sinogram()

    def clear_sinogram(self, event):
        self.state.sino = None
        self.state.theta_rec = []
        self.redraw_sinogram()

    def redraw_sinogram(self):
        if self.state.sino is None:
            data = np.zeros((self.state.n_detectors, 1))
        else:
            data = self.state.sino
        self.sino_im.set_data(data)
        self.sino_im.set_extent([0, data.shape[1], 0, data.shape[0]])
        self.ax_sino.set_xlim(0, max(1, data.shape[1]))
        self.ax_sino.set_ylim(0, data.shape[0])
        self.fig.canvas.draw_idle()

    def do_reconstruct(self, event):
        if self.state.sino is None or len(self.state.theta_rec) < 2:
            print("Not enough data for reconstruction.")
            return
        self.state.recon = reconstruct(
            self.state.sino,
            self.state.theta_rec,
            self.state.filter_name,
            self.state.interpolation,
            self.state.recon_size,
            self.state.mask,
        )
        self.img_im.set_data(self.state.recon)
        if self.state.recon is not None:
            p = psnr(self.state.img, self.state.recon, data_range=1)
            s = ssim(self.state.img, self.state.recon, data_range=1)
            self.ax_img.set_title(
                f"Reconstruction\nPSNR {p:.2f} dB, SSIM {s:.3f}")
        self.fig.canvas.draw_idle()

    def load_image_dialog(self, event):
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(filetypes=[("image", "*.png;*.jpg;*.jpeg")])
            root.destroy()
            if not path:
                return
        except Exception:
            path = input("Path to image: ")
        img = load_image(path, self.state.recon_size)
        self.state.img = img
        self.reset_view(None)
        self.update_projection()

    def save_sinogram(self, event):
        if self.state.sino is None:
            print("Nothing to save")
            return
        save_npz("sinogram.npz", self.state.sino, self.state.theta_rec)
        print("Saved sinogram.npz")

    def load_sinogram(self, event):
        if not os.path.exists("sinogram.npz"):
            print("sinogram.npz not found")
            return
        self.state.sino, self.state.theta_rec = load_npz("sinogram.npz")
        self.state.n_detectors = self.state.sino.shape[0]
        self.tb_detectors.set_val(str(self.state.n_detectors))
        self.redraw_sinogram()

    def export_pngs(self, event):
        export_pngs(self.state.img, self.state.sino, self.state.recon)
        print("Exported source.png, sinogram.png, reconstruction.png")

    def on_filter_change(self, label):
        self.state.filter_name = label

    def on_interp_change(self, label):
        self.state.interpolation = label

    def on_noise_change(self, label):
        self.state.noise_level = label
        self.update_projection()

    def on_show_intensity(self, label):
        self.state.show_intensity = not self.state.show_intensity
        self.update_projection()

    def on_mask_toggle(self, label):
        self.state.mask = not self.state.mask

    def on_detectors_submit(self, text):
        try:
            n = int(float(text))
            if n <= 0:
                raise ValueError
            self.state.n_detectors = n
            self.update_projection()
            self.redraw_sinogram()
        except Exception:
            print("Invalid detector count")

    def on_recon_size_submit(self, text):
        try:
            n = int(float(text))
            if n <= 0:
                raise ValueError
            self.state.recon_size = n
        except Exception:
            print("Invalid size")

    def on_sweep_step_submit(self, text):
        try:
            self.state.sweep_step = float(text)
        except Exception:
            print("Invalid sweep step")

    def on_image_choice(self, label):
        if label == "phantom":
            img = data.shepp_logan_phantom().astype(np.float32)
            img = resize(img, (self.state.recon_size, self.state.recon_size), anti_aliasing=True)
        else:  # smiley
            img = self.generate_smiley(self.state.recon_size)
        self.state.img = img
        self.reset_view(None)
        self.update_projection()

    def reset_view(self, event):
        self.img_im.set_data(self.state.img)
        self.ax_img.set_title("Source / Reconstruction")
        self.fig.canvas.draw_idle()

    # Utility: generate smiley image
    def generate_smiley(self, size):
        img = np.zeros((size, size), dtype=np.float32)
        yy, xx = np.indices((size, size))
        cx = cy = size / 2
        r = size * 0.45
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
        img[mask] = 0.2
        # eyes
        r_eye = r * 0.1
        for ex in (-0.2, 0.2):
            x0 = cx + ex * size
            y0 = cy - 0.1 * size
            mask_eye = (xx - x0) ** 2 + (yy - y0) ** 2 <= r_eye ** 2
            img[mask_eye] = 1.0
        # mouth
        angle = np.arctan2(yy - (cy + 0.1 * size), xx - cx)
        dist = np.hypot(xx - cx, yy - (cy + 0.1 * size))
        mouth = (dist >= r * 0.5) & (dist <= r * 0.6) & (angle > -np.pi / 2) & (
            angle < np.pi / 2
        )
        img[mouth] = 1.0
        return img

    # ------------------------------------------------------------------
    def start(self):
        plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    app = RadonDemo()
    app.start()


if __name__ == "__main__":
    main()