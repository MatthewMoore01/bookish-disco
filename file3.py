"""demo_radon.py - Interactive Radon transform demo.

How to run
----------
python demo_radon.py

This script demonstrates the Radon transform (forward projection),
sinogram acquisition and filtered back–projection reconstruction.
It is intended for teaching the idea behind computed tomography and
computed axial lithography.  The interface uses only matplotlib widgets
and should run on macOS, Linux and Windows with a standard interactive
backend.  Dependencies: numpy, matplotlib, scikit-image, Pillow.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import (
    Slider, Button, CheckButtons, RadioButtons, TextBox
)

from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

# Optional: for file dialog
try:
    import tkinter as tk
    from tkinter import filedialog
    TK_AVAILABLE = True
except Exception:  # pragma: no cover - fail quietly if Tk not available
    TK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def compute_projection(img, theta_deg, n_detectors):
    """Compute 1D line integral (Radon) at angle ``theta_deg``.

    Parameters
    ----------
    img : ndarray
        2D float image in [0,1].
    theta_deg : float
        Angle in degrees.
    n_detectors : int
        Number of detector samples.

    Returns
    -------
    L : ndarray, shape (n_detectors,)
        Line integrals.
    """
    img_res = resize(
        img, (n_detectors, n_detectors), order=1, mode="constant",
        anti_aliasing=True, preserve_range=True
    )
    proj = radon(img_res, [theta_deg], circle=True)
    return proj[:, 0]


def append_measurement(sino_list, theta_list, L, theta_deg):
    """Append or replace a measurement column at ``theta_deg``."""
    if theta_deg in theta_list:
        idx = theta_list.index(theta_deg)
        sino_list[idx] = L
    else:
        theta_list.append(theta_deg)
        sino_list.append(L)


def reconstruct(sino, theta_list, filter_name, interpolation,
                output_size, apply_mask):
    """Perform filtered back–projection."""
    theta_arr = np.array(theta_list)
    order = np.argsort(theta_arr)
    theta_sorted = theta_arr[order]
    sino_sorted = sino[:, order]
    filt = None if filter_name == 'none' else filter_name
    recon = iradon(
        sino_sorted, theta=theta_sorted, filter_name=filt,
        interpolation=interpolation, circle=True, output_size=output_size
    )
    if apply_mask:
        ny, nx = recon.shape
        y, x = np.ogrid[:ny, :nx]
        cy, cx = ny / 2.0, nx / 2.0
        mask = (x - cx)**2 + (y - cy)**2 <= (min(nx, ny)/2.0)**2
        recon = recon * mask
    return recon


def load_image(path, size=None):
    """Load image from ``path`` and return float array in [0,1]."""
    with Image.open(path) as im:
        im = im.convert('L')  # greyscale
        arr = np.asarray(im, dtype=np.float32) / 255.0
    if size is not None:
        arr = resize(arr, size, anti_aliasing=True, preserve_range=True)
    return arr.astype(np.float32)


def save_npz(filename, sino, theta_list):
    np.savez(filename, sino=sino.astype(np.float32),
             theta_deg=np.array(theta_list, dtype=np.float32))


def load_npz(filename):
    data = np.load(filename)
    sino = data['sino']
    theta = data['theta_deg'].tolist()
    return sino, theta


def export_pngs(base, img, sino, recon):
    if img is not None:
        plt.imsave(base + '_source.png', img, cmap='gray')
    if sino is not None and sino.size:
        plt.imsave(base + '_sino.png', sino, cmap='gray')
    if recon is not None:
        plt.imsave(base + '_recon.png', recon, cmap='gray')


def make_smiley(size=256):
    """Generate a simple smiley face image."""
    y, x = np.ogrid[-1:1:size*1j, -1:1:size*1j]
    img = np.zeros((size, size), dtype=np.float32)
    face = x**2 + y**2 <= 1
    img[face] = 0.2
    eyes = (x-0.35)**2 + (y+0.35)**2 <= 0.05
    img[eyes] = 1.0
    eyes = (x+0.35)**2 + (y+0.35)**2 <= 0.05
    img[eyes] = 1.0
    mouth = ((x)**2 + (y-0.25)**2 <= 0.5**2) & (y < 0.1)
    img[mouth] = 1.0
    return img


def choose_initial_image(size=256):
    """Prompt the user to choose the initial image."""
    print('Choose initial image:')
    print('1 - Shepp-Logan phantom (default)')
    print('2 - Smiley')
    print('3 - Load from file')
    choice = input('Selection [1/2/3]: ').strip()
    if choice == '2':
        return make_smiley(size)
    if choice == '3' and TK_AVAILABLE:
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg')])
        root.destroy()
        if path:
            return load_image(path, size=(size, size))
    # default
    img = shepp_logan_phantom().astype(np.float32)
    img = resize(img, (size, size), preserve_range=True)
    return img


# ---------------------------------------------------------------------------
# Demo class
# ---------------------------------------------------------------------------


class CTDemo:
    def __init__(self, img=None):
        if img is None:
            img = shepp_logan_phantom().astype(np.float32)
            img = resize(img, (256, 256), preserve_range=True)
        self.img = img
        self.n_detectors = 384
        self.theta_rec = []
        self.sino_list = []  # list of columns (ndarray)
        self.recon = None
        self.show_intensity = False
        self.noise_sigma = 0.0
        self.sweep_step = 1.0
        self.output_size = self.img.shape[0]
        self.interpolation = 'linear'
        self.filter_name = 'ramp'
        self.apply_mask = True
        self.show_recon = False
        self.playing = False
        self.binary_mode = False
        self.metrics_text = None

        self._init_fig()
        self._init_widgets()
        self.update_projection(self.angle_slider.val)

    # ------------------------------------------------------------------
    def _init_fig(self):
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(2, 3, height_ratios=[10, 1])
        self.ax_sino = self.fig.add_subplot(gs[0, 0])
        self.ax_img = self.fig.add_subplot(gs[0, 1])
        self.ax_proj = self.fig.add_subplot(gs[0, 2])
        self.ax_ctrl = self.fig.add_subplot(gs[1, :])
        self.ax_ctrl.axis('off')

        self.img_artist = self.ax_img.imshow(self.img, cmap='gray', vmin=0, vmax=1)
        self.ax_img.set_title('Source')
        self.ax_img.axis('off')

        self.sino_artist = self.ax_sino.imshow(
            np.zeros((self.n_detectors, 1)), cmap='gray',
            aspect='auto', vmin=0, vmax=1
        )
        self.ax_sino.set_title('Sinogram')
        self.ax_sino.set_xlabel('Angle index')
        self.ax_sino.set_ylabel('Detector')

        self.proj_line, = self.ax_proj.plot([], [])
        self.ax_proj.set_title('Projection')
        self.ax_proj.set_xlabel('Detector bin')
        self.ax_proj.set_ylabel('Line integral')

        # Geometry overlay
        H, W = self.img.shape
        self.centre = (W/2.0, H/2.0)
        self.det_line, = self.ax_img.plot([], [], 'r-')
        self.ray_lines = [self.ax_img.plot([], [], 'r:', lw=0.8)[0]
                          for _ in range(5)]
        self.ax_img.plot(self.centre[0], self.centre[1], 'ro', ms=3)

    # ------------------------------------------------------------------
    def _init_widgets(self):
        ax_angle = self.fig.add_axes([0.1, 0.05, 0.3, 0.03])
        self.angle_slider = Slider(ax_angle, 'Angle [deg]', 0, 180,
                                   valinit=0, valstep=0.5)
        self.angle_slider.on_changed(self.on_angle_change)

        ax_play = self.fig.add_axes([0.42, 0.05, 0.08, 0.04])
        self.play_button = Button(ax_play, 'Play/Pause')
        self.play_button.on_clicked(self.toggle_play)

        ax_rec = self.fig.add_axes([0.52, 0.05, 0.08, 0.04])
        self.rec_button = Button(ax_rec, 'Record current')
        self.rec_button.on_clicked(self.record_current)

        ax_sweep = self.fig.add_axes([0.62, 0.05, 0.08, 0.04])
        self.sweep_button = Button(ax_sweep, 'Record sweep')
        self.sweep_button.on_clicked(self.record_sweep)

        ax_clear = self.fig.add_axes([0.72, 0.05, 0.08, 0.04])
        self.clear_button = Button(ax_clear, 'Clear sino')
        self.clear_button.on_clicked(self.clear_sino)

        ax_recon = self.fig.add_axes([0.82, 0.05, 0.08, 0.04])
        self.recon_button = Button(ax_recon, 'Reconstruct')
        self.recon_button.on_clicked(self.do_recon)

        # Second row of widgets
        ax_load = self.fig.add_axes([0.1, 0.005, 0.08, 0.04])
        self.load_img_button = Button(ax_load, 'Load image')
        self.load_img_button.on_clicked(self.load_image_dialog)

        ax_save = self.fig.add_axes([0.20, 0.005, 0.08, 0.04])
        self.save_sino_button = Button(ax_save, 'Save sino')
        self.save_sino_button.on_clicked(self.save_sino)

        ax_loads = self.fig.add_axes([0.30, 0.005, 0.08, 0.04])
        self.load_sino_button = Button(ax_loads, 'Load sino')
        self.load_sino_button.on_clicked(self.load_sino)

        ax_export = self.fig.add_axes([0.40, 0.005, 0.08, 0.04])
        self.export_button = Button(ax_export, 'Export PNGs')
        self.export_button.on_clicked(self.export_pngs)

        # Filter radio buttons
        ax_filter = self.fig.add_axes([0.5, 0.005, 0.1, 0.08])
        self.filter_radio = RadioButtons(
            ax_filter,
            ('ramp', 'shepp-logan', 'hann', 'hamming', 'cosine', 'none'))
        self.filter_radio.on_clicked(self.set_filter)

        # Interpolation radio buttons
        ax_interp = self.fig.add_axes([0.62, 0.005, 0.1, 0.05])
        self.interp_radio = RadioButtons(ax_interp, ('nearest', 'linear'))
        self.interp_radio.on_clicked(self.set_interpolation)

        # Show intensity checkbox
        ax_int = self.fig.add_axes([0.74, 0.005, 0.08, 0.05])
        self.intensity_chk = CheckButtons(ax_int, ['Intensity'], [False])
        self.intensity_chk.on_clicked(self.toggle_intensity)

        # Noise level radio
        ax_noise = self.fig.add_axes([0.84, 0.005, 0.1, 0.08])
        self.noise_radio = RadioButtons(
            ax_noise, ('none', 'low', 'medium', 'high'))
        self.noise_radio.on_clicked(self.set_noise)

        # Text boxes for numeric controls
        ax_det = self.fig.add_axes([0.1, 0.11, 0.07, 0.04])
        self.det_box = TextBox(ax_det, 'Detectors', initial=str(self.n_detectors))
        self.det_box.on_submit(self.set_detectors)

        ax_out = self.fig.add_axes([0.19, 0.11, 0.07, 0.04])
        self.out_box = TextBox(ax_out, 'Recon size', initial=str(self.output_size))
        self.out_box.on_submit(self.set_output_size)

        ax_step = self.fig.add_axes([0.28, 0.11, 0.07, 0.04])
        self.step_box = TextBox(ax_step, 'Sweep step', initial=str(self.sweep_step))
        self.step_box.on_submit(self.set_sweep_step)

        # Show recon toggle and mask checkbox
        ax_show = self.fig.add_axes([0.37, 0.11, 0.08, 0.05])
        self.show_chk = CheckButtons(ax_show, ['Show recon'], [False])
        self.show_chk.on_clicked(self.toggle_recon)

        ax_mask = self.fig.add_axes([0.47, 0.11, 0.08, 0.05])
        self.mask_chk = CheckButtons(ax_mask, ['Mask'], [True])
        self.mask_chk.on_clicked(self.toggle_mask)

        # Reset view button
        ax_reset = self.fig.add_axes([0.57, 0.11, 0.07, 0.04])
        self.reset_button = Button(ax_reset, 'Reset view')
        self.reset_button.on_clicked(self.reset_view)

        # Undersampling preset buttons
        self.preset_axes = []
        self.preset_buttons = []
        presets = [8, 16, 32, 64, 180]
        for i, p in enumerate(presets):
            axp = self.fig.add_axes([0.60 + 0.06*i, 0.11, 0.05, 0.04])
            bp = Button(axp, f'{p}°')
            bp.on_clicked(lambda event, n=p: self.apply_preset(n))
            self.preset_axes.append(axp)
            self.preset_buttons.append(bp)

        # Compare filters button
        ax_cmp = self.fig.add_axes([0.96, 0.11, 0.04, 0.04])
        self.cmp_button = Button(ax_cmp, 'Cmp')
        self.cmp_button.on_clicked(self.compare_filters)

        # Binary checkbox for loaded images
        ax_bin = self.fig.add_axes([0.90, 0.11, 0.06, 0.05])
        self.bin_chk = CheckButtons(ax_bin, ['Binary'], [False])
        self.bin_chk.on_clicked(self.toggle_binary)

        # Timer for play
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self._timer_step)

    # ------------------------------------------------------------------
    # Widget callbacks

    def on_angle_change(self, val):
        self.update_projection(val)

    def toggle_play(self, event):
        self.playing = not self.playing
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _timer_step(self):
        a = self.angle_slider.val + 1
        if a > 180:
            a -= 180
        self.angle_slider.set_val(a)

    def record_current(self, event=None):
        theta = float(self.angle_slider.val)
        L = compute_projection(self.img, theta, self.n_detectors)
        append_measurement(self.sino_list, self.theta_rec, L, theta)
        self.update_sino_plot()
        self.update_projection(theta)

    def record_sweep(self, event=None):
        start = float(self.angle_slider.val)
        step = self.sweep_step
        for theta in np.arange(start, 180.0 + 1e-3, step):
            L = compute_projection(self.img, theta, self.n_detectors)
            append_measurement(self.sino_list, self.theta_rec, L, theta)
        self.update_sino_plot()
        self.update_projection(theta)
        self.angle_slider.set_val(theta % 180)

    def clear_sino(self, event=None):
        self.sino_list = []
        self.theta_rec = []
        self.update_sino_plot()
        self.recon = None
        self.reset_view()

    def do_recon(self, event=None):
        if len(self.theta_rec) < 2:
            print('Need at least two angles')
            return
        sino = np.column_stack(self.sino_list)
        recon = reconstruct(sino, self.theta_rec, self.filter_name,
                             self.interpolation, self.output_size,
                             self.apply_mask)
        self.recon = recon
        self.show_recon = True
        self.show_chk.set_active(0)
        self.update_image()
        if self.metrics_text:
            self.metrics_text.remove()
        psnr = peak_signal_noise_ratio(self.img, recon, data_range=1)
        ssim = structural_similarity(self.img, recon, data_range=1)
        self.metrics_text = self.ax_img.text(
            0.02, 0.02, f'PSNR: {psnr:.2f}\nSSIM: {ssim:.3f}',
            transform=self.ax_img.transAxes, color='yellow',
            bbox=dict(boxstyle='round', fc='black', alpha=0.5),
            fontsize=8
        )
        self.fig.canvas.draw_idle()

    def load_image_dialog(self, event=None):
        if not TK_AVAILABLE:
            print('Tkinter not available')
            return
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg')])
        root.destroy()
        if path:
            arr = load_image(path, size=self.img.shape)
            if self.binary_mode:
                arr = (arr > 0.5).astype(np.float32)
            self.img = arr
            self.output_size = self.img.shape[0]
            self.out_box.set_val(str(self.output_size))
            self.clear_sino()
            self.update_image()

    def save_sino(self, event=None):
        if not self.sino_list:
            return
        sino = np.column_stack(self.sino_list)
        save_npz('sinogram.npz', sino, self.theta_rec)

    def load_sino(self, event=None):
        try:
            sino, theta = load_npz('sinogram.npz')
        except Exception as e:
            print('Failed to load sinogram:', e)
            return
        self.sino_list = [sino[:, i] for i in range(sino.shape[1])]
        self.theta_rec = list(theta)
        self.n_detectors = sino.shape[0]
        self.det_box.set_val(str(self.n_detectors))
        self.update_sino_plot()

    def export_pngs(self, event=None):
        sino = np.column_stack(self.sino_list) if self.sino_list else None
        export_pngs('ct_demo', self.img, sino, self.recon)

    def set_filter(self, label):
        self.filter_name = label

    def set_interpolation(self, label):
        self.interpolation = label

    def toggle_intensity(self, labels):
        self.show_intensity = not self.show_intensity
        self.update_projection(self.angle_slider.val)

    def set_noise(self, label):
        mapping = {'none': 0.0, 'low': 0.01, 'medium': 0.05, 'high': 0.1}
        self.noise_sigma = mapping.get(label, 0.0)
        self.update_projection(self.angle_slider.val)

    def set_detectors(self, text):
        try:
            val = int(text)
            if val > 0:
                self.n_detectors = val
                self.clear_sino()
                self.update_projection(self.angle_slider.val)
        except ValueError:
            pass

    def set_output_size(self, text):
        try:
            val = int(text)
            if val > 0:
                self.output_size = val
        except ValueError:
            pass

    def set_sweep_step(self, text):
        try:
            self.sweep_step = float(text)
        except ValueError:
            pass

    def toggle_recon(self, labels):
        self.show_recon = not self.show_recon
        self.update_image()

    def toggle_mask(self, labels):
        self.apply_mask = not self.apply_mask

    def toggle_binary(self, labels):
        self.binary_mode = not self.binary_mode

    def reset_view(self, event=None):
        self.show_recon = False
        if self.metrics_text:
            self.metrics_text.remove()
            self.metrics_text = None
        self.show_chk.set_active(0) if self.show_chk.get_status()[0] else None
        self.update_image()

    def apply_preset(self, n_angles):
        self.clear_sino()
        angles = np.linspace(0, 180, n_angles, endpoint=False)
        for theta in angles:
            L = compute_projection(self.img, theta, self.n_detectors)
            append_measurement(self.sino_list, self.theta_rec, L, float(theta))
        self.update_sino_plot()

    def compare_filters(self, event=None):
        if len(self.theta_rec) < 2:
            print('Need data for comparison')
            return
        sino = np.column_stack(self.sino_list)
        filters = ['ramp', 'shepp-logan', 'hann']
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for ax, f in zip(axes, filters):
            r = reconstruct(sino, self.theta_rec, f,
                             self.interpolation, self.output_size,
                             self.apply_mask)
            psnr = peak_signal_noise_ratio(self.img, r, data_range=1)
            ssim = structural_similarity(self.img, r, data_range=1)
            ax.imshow(r, cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'{f}\nPSNR {psnr:.1f}\nSSIM {ssim:.3f}')
            ax.axis('off')
        fig.suptitle('Filter comparison')
        fig.show()

    # ------------------------------------------------------------------
    # Update functions

    def update_image(self):
        if self.show_recon and self.recon is not None:
            self.img_artist.set_data(self.recon)
            self.ax_img.set_title('Reconstruction')
        else:
            self.img_artist.set_data(self.img)
            self.ax_img.set_title('Source')
        self.fig.canvas.draw_idle()

    def update_sino_plot(self):
        if self.sino_list:
            sino = np.column_stack(self.sino_list)
        else:
            sino = np.zeros((self.n_detectors, 1))
        self.sino_artist.set_data(sino)
        self.sino_artist.set_extent([0, sino.shape[1], 0, self.n_detectors])
        self.ax_sino.set_ylim(self.n_detectors, 0)
        self.fig.canvas.draw_idle()

    def update_projection(self, theta):
        L = compute_projection(self.img, float(theta), self.n_detectors)
        if self.show_intensity:
            I = np.exp(-L)
            if self.noise_sigma > 0:
                I = np.clip(I + np.random.normal(0, self.noise_sigma, size=I.shape), 0, 1)
            y = I
            self.ax_proj.set_ylabel('Intensity')
        else:
            y = L
            self.ax_proj.set_ylabel('Line integral')
        x = np.arange(len(y))
        self.proj_line.set_data(x, y)
        self.ax_proj.set_xlim(0, len(y))
        self.ax_proj.relim()
        self.ax_proj.autoscale_view(True, True, True)
        self.update_geometry(theta)
        self.fig.canvas.draw_idle()

    def update_geometry(self, theta):
        H, W = self.img.shape
        cx, cy = self.centre
        t = np.deg2rad(theta)
        d = max(H, W)
        dirx, diry = np.cos(t), np.sin(t)
        # Detector axis line
        x0 = cx - dirx * d
        x1 = cx + dirx * d
        y0 = cy - diry * d
        y1 = cy + diry * d
        self.det_line.set_data([x0, x1], [y0, y1])
        # Rays
        perp = (-diry, dirx)
        offsets = np.linspace(-d/4, d/4, len(self.ray_lines))
        for off, line in zip(offsets, self.ray_lines):
            bx = cx + dirx*off
            by = cy + diry*off
            rx0 = bx - perp[0]*d
            ry0 = by - perp[1]*d
            rx1 = bx + perp[0]*d
            ry1 = by + perp[1]*d
            line.set_data([rx0, rx1], [ry0, ry1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    img = choose_initial_image(256)
    demo = CTDemo(img=img)
    plt.show()


if __name__ == '__main__':
    main()