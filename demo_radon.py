"""Radon transform teaching demo.

This script demonstrates forward projection (Radon transform), sinogram
collection and filtered back-projection reconstruction with a small GUI
based on matplotlib widgets.  It is intended for exploring the principles of
computerised tomography and computed axial lithography.

How to run
==========
    python demo_radon.py

Dependencies: numpy, matplotlib, scikit-image, Pillow.

"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import (Slider, Button, RadioButtons, CheckButtons,
                                TextBox)
from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

# ---------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------

def compute_projection(img, theta_deg, n_detectors):
    """Return line integral of ``img`` at angle ``theta_deg``.

    Parameters
    ----------
    img : ndarray
        2-D image in [0, 1].  Will be resized to ``(n_detectors, n_detectors)``
        for the transform.
    theta_deg : float
        Angle in degrees.
    n_detectors : int
        Number of detector samples.

    Returns
    -------
    L : ndarray, shape (n_detectors,)
        Line integrals returned by :func:`skimage.transform.radon`.
    """
    img_r = resize(img, (n_detectors, n_detectors), order=1,
                   preserve_range=True, anti_aliasing=True)
    sino = radon(img_r, [theta_deg], circle=True)
    return np.asarray(sino[:, 0], dtype=np.float32)


def append_measurement(sino, theta_list, theta, L):
    """Append or replace a measurement in ``sino`` at angle ``theta``.

    ``sino`` is a 2-D array with shape (n_detectors, n_angles).  ``theta_list``
    is a Python list of recorded angles in degrees.
    """
    if theta in theta_list:
        idx = theta_list.index(theta)
        sino[:, idx] = L
    else:
        theta_list.append(theta)
        if sino.size == 0:
            sino = L[:, None]
        else:
            sino = np.hstack([sino, L[:, None]])
    return sino, theta_list


def reconstruct(sino, theta_list, filter_name, interpolation, output_size,
                mask=True):
    """Perform filtered back projection."""
    if sino.size == 0 or len(theta_list) < 2:
        return None
    theta_arr = np.asarray(theta_list, dtype=float)
    order = np.argsort(theta_arr)
    theta_arr = theta_arr[order]
    sino = sino[:, order]
    filt = None if filter_name == 'none' else filter_name
    recon = iradon(sino, theta=theta_arr, circle=True,
                   filter_name=filt, interpolation=interpolation,
                   output_size=output_size)
    if mask:
        # Apply circular mask
        h, w = recon.shape
        Y, X = np.ogrid[:h, :w]
        centre = (h/2.0, w/2.0)
        r = min(centre)
        mask_arr = (X-centre[1])**2 + (Y-centre[0])**2 <= r*r
        recon = recon * mask_arr
    return recon.astype(np.float32)


def load_image(path):
    """Load an image file as greyscale float32 array in [0,1]."""
    img = Image.open(path).convert('L')
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def save_npz(path, sino, theta_deg):
    np.savez(path, sino=np.asarray(sino, dtype=np.float32),
             theta_deg=np.asarray(theta_deg, dtype=np.float32))


def load_npz(path):
    data = np.load(path)
    sino = data['sino'].astype(np.float32)
    theta = list(data['theta_deg'].astype(float))
    return sino, theta


def export_pngs(base_path, img, sino, recon):
    plt.imsave(base_path + '_image.png', img, cmap='gray')
    if sino.size:
        plt.imsave(base_path + '_sinogram.png', sino, cmap='gray')
    if recon is not None:
        plt.imsave(base_path + '_recon.png', recon, cmap='gray')

# ---------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------

def make_smiley(size):
    """Generate a simple smiley face."""
    img = np.zeros((size, size), dtype=np.float32)
    Y, X = np.ogrid[:size, :size]
    centre = size / 2
    mask = (X-centre)**2 + (Y-centre)**2 <= (0.48*size)**2
    img[mask] = 0.2
    eye_r = 0.07 * size
    for ex in (-0.15*size, 0.15*size):
        mask = (X-(centre+ex))**2 + (Y-(centre-0.1*size))**2 <= eye_r**2
        img[mask] = 1.0
    # smile
    r_smile = 0.3 * size
    mask = ((X-centre)**2 + (Y-(centre+0.1*size))**2 <= r_smile**2) & \
           (Y > centre)
    img[mask] = 1.0
    img = np.clip(img, 0, 1)
    return img

# ---------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------

class RadonDemo:
    def __init__(self, img):
        self.img = img.astype(np.float32)
        self.recon = None
        self.theta_rec = []
        self.sino = np.zeros((384, 0), dtype=np.float32)

        # Parameters
        self.n_detectors = 384
        self.output_size = img.shape[0]
        self.sweep_step = 1.0
        self.show_intensity = False
        self.noise_level = 'none'
        self.filter_name = 'ramp'
        self.interp = 'linear'
        self.mask_on = True

        # Setup figure and axes
        self.fig, (self.ax_sino, self.ax_img, self.ax_proj) = \
            plt.subplots(1, 3, figsize=(12, 4))
        plt.subplots_adjust(bottom=0.32)

        self.ax_sino.set_title('Sinogram')
        self.ax_img.set_title('Source')
        self.ax_proj.set_title('Projection')

        self.img_artist = self.ax_img.imshow(self.img, cmap='gray',
                                             vmin=0, vmax=1)
        self.sino_artist = self.ax_sino.imshow(np.zeros((self.n_detectors, 1)),
                                               cmap='gray', aspect='auto',
                                               vmin=0, vmax=1)
        self.proj_line, = self.ax_proj.plot([], [], lw=1)
        self.ax_proj.set_ylim(0, 1)
        self.ax_proj.set_xlim(0, self.n_detectors)

        # Geometry overlay lines
        self.det_line, = self.ax_img.plot([], [], 'r-')
        self.ray_lines = [self.ax_img.plot([], [], 'r-', lw=0.5)[0]
                          for _ in range(3)]

        # Widgets ------------------------------------------------
        ax_angle = plt.axes([0.1, 0.27, 0.6, 0.03])
        self.slider_angle = Slider(ax_angle, 'Angle [deg]', 0, 180, valinit=0,
                                   valstep=0.5)
        self.slider_angle.on_changed(self.on_angle_change)

        ax_play = plt.axes([0.75, 0.27, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.on_play)
        self.animating = False
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self._animate)

        # Button row 1
        self.btn_record = Button(plt.axes([0.05, 0.20, 0.1, 0.04]),
                                 'Record current')
        self.btn_record.on_clicked(self.on_record)
        self.btn_sweep = Button(plt.axes([0.17, 0.20, 0.1, 0.04]),
                                'Record sweep')
        self.btn_sweep.on_clicked(self.on_sweep)
        self.btn_clear = Button(plt.axes([0.29, 0.20, 0.1, 0.04]),
                                'Clear sino')
        self.btn_clear.on_clicked(self.on_clear)
        self.btn_recon = Button(plt.axes([0.41, 0.20, 0.1, 0.04]),
                                'Reconstruct')
        self.btn_recon.on_clicked(self.on_reconstruct)

        # Button row 2
        self.btn_load_img = Button(plt.axes([0.05, 0.14, 0.1, 0.04]),
                                   'Load image')
        self.btn_load_img.on_clicked(self.on_load_image)
        self.btn_save_sino = Button(plt.axes([0.17, 0.14, 0.1, 0.04]),
                                    'Save sino')
        self.btn_save_sino.on_clicked(self.on_save_sino)
        self.btn_load_sino = Button(plt.axes([0.29, 0.14, 0.1, 0.04]),
                                    'Load sino')
        self.btn_load_sino.on_clicked(self.on_load_sino)
        self.btn_export = Button(plt.axes([0.41, 0.14, 0.1, 0.04]),
                                 'Export PNGs')
        self.btn_export.on_clicked(self.on_export_pngs)

        # Button row 3
        self.btn_toggle_view = Button(plt.axes([0.53, 0.20, 0.1, 0.04]),
                                      'Toggle view')
        self.btn_toggle_view.on_clicked(self.on_toggle_view)
        self.btn_reset_view = Button(plt.axes([0.53, 0.14, 0.1, 0.04]),
                                     'Reset view')
        self.btn_reset_view.on_clicked(self.on_reset_view)

        # Radio buttons for filter and interpolation
        ax_filter = plt.axes([0.66, 0.18, 0.1, 0.12])
        self.radio_filter = RadioButtons(ax_filter,
                                         ('ramp', 'shepp-logan', 'hann',
                                          'hamming', 'cosine', 'none'))
        self.radio_filter.on_clicked(self.on_filter_change)

        ax_interp = plt.axes([0.78, 0.18, 0.1, 0.07])
        self.radio_interp = RadioButtons(ax_interp, ('linear', 'nearest'))
        self.radio_interp.on_clicked(self.on_interp_change)

        ax_noise = plt.axes([0.90, 0.18, 0.08, 0.12])
        self.radio_noise = RadioButtons(ax_noise,
                                        ('none', 'low', 'medium', 'high'))
        self.radio_noise.on_clicked(self.on_noise_change)

        # Check for intensity / mask
        ax_check = plt.axes([0.66, 0.14, 0.1, 0.04])
        self.check_intensity = CheckButtons(ax_check, ['Show intensity'],
                                            [self.show_intensity])
        self.check_intensity.on_clicked(self.on_intensity_toggle)

        ax_mask = plt.axes([0.78, 0.14, 0.1, 0.04])
        self.check_mask = CheckButtons(ax_mask, ['Mask'], [self.mask_on])
        self.check_mask.on_clicked(self.on_mask_toggle)

        # Numeric controls
        self.txt_det = TextBox(plt.axes([0.90, 0.14, 0.08, 0.04]),
                               'Detectors', initial=str(self.n_detectors))
        self.txt_det.on_submit(self.on_detectors_change)
        self.txt_recon = TextBox(plt.axes([0.90, 0.09, 0.08, 0.04]),
                                 'Recon size', initial=str(self.output_size))
        self.txt_recon.on_submit(self.on_reconsize_change)
        self.txt_step = TextBox(plt.axes([0.90, 0.04, 0.08, 0.04]),
                                'Sweep step', initial=str(self.sweep_step))
        self.txt_step.on_submit(self.on_step_change)

        # Metrics text
        self.metric_text = self.fig.text(0.55, 0.02, '')

        self.on_angle_change(0)

    # -------------------------------------------------------
    # Widget callbacks
    # -------------------------------------------------------
    def on_angle_change(self, val):
        theta = float(self.slider_angle.val)
        L = compute_projection(self.img, theta, self.n_detectors)
        x = np.arange(self.n_detectors)
        if self.show_intensity:
            I = np.exp(-L)
            sigma = {'none':0, 'low':0.01, 'medium':0.05, 'high':0.1}[self.noise_level]
            if sigma>0:
                I = np.clip(I + np.random.normal(0, sigma, size=I.shape),0,1)
            self.proj_line.set_data(x, I)
            self.ax_proj.set_ylim(0, 1)
            self.ax_proj.set_ylabel('Intensity')
        else:
            self.proj_line.set_data(x, L)
            self.ax_proj.set_ylim(L.min(), L.max())
            self.ax_proj.set_ylabel('Line integral')
        self.ax_proj.set_xlim(0, self.n_detectors)
        self.fig.canvas.draw_idle()
        self._update_overlay(theta)

    def on_play(self, event):
        self.animating = not self.animating
        self.btn_play.label.set_text('Pause' if self.animating else 'Play')
        if self.animating:
            self.timer.start()
        else:
            self.timer.stop()

    def _animate(self):
        ang = self.slider_angle.val + 1
        if ang > 180:
            ang -= 180
        self.slider_angle.set_val(ang)

    def on_record(self, event):
        theta = float(self.slider_angle.val)
        L = compute_projection(self.img, theta, self.n_detectors)
        self.sino, self.theta_rec = append_measurement(
            self.sino, self.theta_rec, theta, L)
        self._redraw_sino()

    def on_sweep(self, event):
        start = float(self.slider_angle.val)
        angles = np.arange(start, 180+1e-6, float(self.sweep_step))
        for i, th in enumerate(angles):
            L = compute_projection(self.img, th, self.n_detectors)
            self.sino, self.theta_rec = append_measurement(
                self.sino, self.theta_rec, float(th), L)
            if i % 5 == 0:
                self.slider_angle.set_val(th)
                plt.pause(0.001)
        self._redraw_sino()

    def on_clear(self, event):
        self.sino = np.zeros((self.n_detectors, 0), dtype=np.float32)
        self.theta_rec = []
        self._redraw_sino()

    def on_reconstruct(self, event):
        self.recon = reconstruct(self.sino, self.theta_rec,
                                 self.filter_name, self.interp,
                                 self.output_size, mask=self.mask_on)
        if self.recon is not None:
            psnr = peak_signal_noise_ratio(self.img, self.recon, data_range=1)
            ssim = structural_similarity(self.img, self.recon, data_range=1)
            self.metric_text.set_text(f'PSNR: {psnr:.2f} dB  SSIM: {ssim:.3f}')
            self.ax_img.set_title('Reconstruction')
            self.img_artist.set_data(self.recon)
            self.fig.canvas.draw_idle()

    def on_load_image(self, event):
        path = input('Path to image file: ').strip()
        if not path:
            return
        if not os.path.exists(path):
            print('File not found')
            return
        self.img = load_image(path)
        self.output_size = self.img.shape[0]
        self.txt_recon.set_val(str(self.output_size))
        self.img_artist.set_data(self.img)
        self.ax_img.set_title('Source')
        self.recon = None
        self.metric_text.set_text('')
        self.on_clear(None)
        self.fig.canvas.draw_idle()

    def on_save_sino(self, event):
        save_npz('sinogram.npz', self.sino, self.theta_rec)
        print('Saved sinogram.npz')

    def on_load_sino(self, event):
        if not os.path.exists('sinogram.npz'):
            print('sinogram.npz not found')
            return
        self.sino, self.theta_rec = load_npz('sinogram.npz')
        self.n_detectors = self.sino.shape[0]
        self.txt_det.set_val(str(self.n_detectors))
        self._redraw_sino()
        print('Loaded sinogram.npz')

    def on_export_pngs(self, event):
        export_pngs('output', self.img, self.sino, self.recon)
        print('Exported PNGs as output_*')

    def on_toggle_view(self, event):
        if self.ax_img.get_title() == 'Source' and self.recon is not None:
            self.ax_img.set_title('Reconstruction')
            self.img_artist.set_data(self.recon)
        else:
            self.ax_img.set_title('Source')
            self.img_artist.set_data(self.img)
        self.fig.canvas.draw_idle()

    def on_reset_view(self, event):
        self.ax_img.set_title('Source')
        self.img_artist.set_data(self.img)
        self.fig.canvas.draw_idle()

    def on_filter_change(self, label):
        self.filter_name = label

    def on_interp_change(self, label):
        self.interp = label

    def on_noise_change(self, label):
        self.noise_level = label

    def on_intensity_toggle(self, label):
        self.show_intensity = not self.show_intensity
        self.on_angle_change(self.slider_angle.val)

    def on_mask_toggle(self, label):
        self.mask_on = not self.mask_on

    def on_detectors_change(self, text):
        try:
            val = int(text)
            if val > 8:
                self.n_detectors = val
                self.on_clear(None)
        except ValueError:
            pass

    def on_reconsize_change(self, text):
        try:
            val = int(text)
            if val > 8:
                self.output_size = val
        except ValueError:
            pass

    def on_step_change(self, text):
        try:
            val = float(text)
            if val > 0:
                self.sweep_step = val
        except ValueError:
            pass

    # -------------------------------------------------------
    def _redraw_sino(self):
        if self.sino.size == 0:
            self.sino_artist.set_data(np.zeros((self.n_detectors, 1)))
        else:
            self.sino_artist.set_data(self.sino)
        self.sino_artist.set_extent([0, max(len(self.theta_rec), 1),
                                     0, self.n_detectors])
        self.fig.canvas.draw_idle()

    def _update_overlay(self, theta):
        h, w = self.img.shape
        cx, cy = w/2, h/2
        length = max(h, w)
        ang = np.deg2rad(theta)
        nx, ny = np.cos(ang), np.sin(ang)
        # detector axis is orthogonal to projection direction
        dx, dy = -ny, nx
        x0, y0 = cx - dx*length, cy - dy*length
        x1, y1 = cx + dx*length, cy + dy*length
        self.det_line.set_data([x0, x1], [y0, y1])
        offsets = np.linspace(-0.4*length, 0.4*length, 3)
        for off, line in zip(offsets, self.ray_lines):
            rx0, ry0 = cx + dx*off - nx*length, cy + dy*off - ny*length
            rx1, ry1 = cx + dx*off + nx*length, cy + dy*off + ny*length
            line.set_data([rx0, rx1], [ry0, ry1])
        self.fig.canvas.draw_idle()

# ---------------------------------------------------------------

def choose_initial_image():
    print('Select initial image:')
    print('1) Shepp-Logan phantom')
    print('2) Smiley')
    print('3) Load from file')
    choice = input('Choice [1]: ').strip() or '1'
    if choice == '1':
        img = shepp_logan_phantom().astype(np.float32)
        img = resize(img, (256, 256), preserve_range=True)
        return img
    elif choice == '2':
        return make_smiley(256)
    elif choice == '3':
        path = input('Path: ').strip()
        if os.path.exists(path):
            return load_image(path)
        else:
            print('File not found, using phantom.')
            img = shepp_logan_phantom().astype(np.float32)
            return resize(img, (256, 256), preserve_range=True)
    else:
        img = shepp_logan_phantom().astype(np.float32)
        img = resize(img, (256, 256), preserve_range=True)
        return img


def main():
    try:
        img = choose_initial_image()
    except Exception as e:
        print('Error loading image:', e)
        return
    app = RadonDemo(img)
    plt.show()


if __name__ == '__main__':
    main()
