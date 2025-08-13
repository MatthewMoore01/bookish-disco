"""
Interactive Radon transform demo for teaching computed tomography and volumetric
printing concepts. A source image is forward projected at varying angles to build a
sinogram, which can then be reconstructed with filtered back-projection. The user
can step through angles, record projections, and reconstruct using different
filters and interpolation schemes.

How to run
----------
    python demo_radon.py
The script opens a single matplotlib window. Use the controls below the images to
record projections and reconstruct.
"""
import argparse
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, TextBox
from skimage.transform import radon, iradon, resize
from skimage.data import shepp_logan_phantom
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.draw import disk
from PIL import Image
import imageio

# ----------------------------- Utility functions -----------------------------

def circular_mask(size):
    """Return a circular mask for a square image of given size."""
    y, x = np.ogrid[-1:1:complex(0, size), -1:1:complex(0, size)]
    mask = x**2 + y**2 <= 1
    return mask.astype(np.float32)


def make_smiley(size=256):
    """Generate a simple smiley face."""
    img = np.zeros((size, size), dtype=np.float32)
    rr, cc = disk((size/2, size/2), size/2.2, shape=img.shape)
    img[rr, cc] = 0.2
    rr, cc = disk((size/3, size/3), size/10, shape=img.shape)
    img[rr, cc] = 1.0
    rr, cc = disk((size/3, size - size/3), size/10, shape=img.shape)
    img[rr, cc] = 1.0
    rr, cc = disk((2*size/3, size/2), size/4, shape=img.shape)
    img[rr, cc] = 1.0
    return img


def load_image(path, size=256, binary=False):
    """Load an image from disk, convert to greyscale float in [0,1]."""
    im = Image.open(path).convert('L')
    im = np.array(im, dtype=np.float32) / 255.0
    im = resize(im, (size, size), anti_aliasing=True)
    if binary:
        im = (im > 0.5).astype(np.float32)
    return im


def save_npz(path, sino, theta):
    np.savez(path, sino=sino.astype(np.float32), theta_deg=np.array(theta, dtype=np.float32))


def load_npz(path):
    data = np.load(path)
    return data['sino'].astype(np.float32), data['theta_deg'].astype(np.float32).tolist()


def export_pngs(base, img, sino, recon=None):
    imageio.imwrite(base + '_image.png', (img * 255).astype(np.uint8))
    if sino.size:
        imageio.imwrite(base + '_sinogram.png', normalise_to_uint8(sino))
    if recon is not None:
        imageio.imwrite(base + '_recon.png', normalise_to_uint8(recon))


def normalise_to_uint8(arr):
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return (arr * 255).astype(np.uint8)


def compute_projection(img, theta_deg, n_detectors):
    """Return line integral for a single angle."""
    L = radon(img, [theta_deg], circle=True)
    L = L[:, 0]
    if len(L) != n_detectors:
        x_old = np.linspace(0, 1, len(L))
        x_new = np.linspace(0, 1, n_detectors)
        L = np.interp(x_new, x_old, L)
    return L.astype(np.float32)


def append_measurement(sino, theta_list, L, theta):
    """Append or replace measurement in the sinogram."""
    if sino is None or sino.size == 0:
        sino = np.zeros((len(L), 0), dtype=np.float32)
    theta = float(theta)
    for i, t in enumerate(theta_list):
        if abs(t - theta) < 1e-3:
            sino[:, i] = L
            theta_list[i] = theta
            break
    else:
        sino = np.column_stack((sino, L.astype(np.float32)))
        theta_list.append(theta)
    return sino, theta_list


def reconstruct(sino, theta_list, filter_name='ramp', interpolation='linear', output_size=None):
    if sino is None or sino.size == 0 or len(theta_list) < 2:
        return None
    order = np.argsort(theta_list)
    theta_arr = np.array(theta_list)[order]
    sino = sino[:, order]
    filt = None if filter_name == 'none' else filter_name
    recon = iradon(sino, theta=theta_arr, circle=True, filter_name=filt,
                   interpolation=interpolation, output_size=output_size)
    return recon.astype(np.float32)

# ------------------------------ Main demo class ------------------------------

class CTDemo:
    def __init__(self):
        self.size = 256
        self.img = shepp_logan_phantom()
        self.img = resize(self.img, (self.size, self.size), anti_aliasing=True).astype(np.float32)
        self.mask = circular_mask(self.size)
        self.img *= self.mask

        self.n_detectors = 384
        self.theta = 0.0
        self.theta_rec = []
        self.sino = np.zeros((self.n_detectors, 0), dtype=np.float32)
        self.recon = None
        self.current_L = compute_projection(self.img, self.theta, self.n_detectors)

        self.noise_sigma = 0.0
        self.show_intensity = False
        self.filter = 'ramp'
        self.interp = 'linear'
        self.output_size = self.size
        self.sweep_step = 1.0
        self.playing = False

        self._build_figure()
        self._update_all()

    # --------------------------- Figure and widgets -----------------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(12, 6))
        gs = self.fig.add_gridspec(1, 3, left=0.05, right=0.98, top=0.95, bottom=0.35, wspace=0.3)
        self.ax_sino = self.fig.add_subplot(gs[0, 0])
        self.ax_img = self.fig.add_subplot(gs[0, 1])
        self.ax_proj = self.fig.add_subplot(gs[0, 2])

        # Images and plots
        self.sino_im = self.ax_sino.imshow(np.zeros((self.n_detectors, 1)), cmap='gray', aspect='auto')
        self.ax_sino.set_title('Sinogram')
        self.ax_sino.set_xlabel('Angle index')
        self.ax_sino.set_ylabel('Detector')

        self.img_im = self.ax_img.imshow(self.img, cmap='gray', vmin=0, vmax=1)
        self.ax_img.set_title('Source / Reconstruction')
        self.ax_img.set_axis_off()

        x = np.arange(self.n_detectors)
        self.proj_line, = self.ax_proj.plot(x, self.current_L)
        self.ax_proj.set_title('Projection at θ = 0°')
        self.ax_proj.set_xlabel('Detector')
        self.ax_proj.set_ylabel('L')

        # Geometry lines
        self.det_line, = self.ax_img.plot([], [], 'r-')
        self.ray_lines = [self.ax_img.plot([], [], 'r:', lw=0.5)[0] for _ in range(3)]

        # Widgets
        ax_slider = self.fig.add_axes([0.15, 0.25, 0.7, 0.03])
        self.slider = Slider(ax_slider, 'Angle [deg]', 0.0, 180.0, valinit=0.0, valstep=0.5)
        self.slider.on_changed(self.on_slider)

        ax_play = self.fig.add_axes([0.05, 0.25, 0.05, 0.03])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.on_play)

        ax_record = self.fig.add_axes([0.05, 0.18, 0.1, 0.04])
        self.btn_record = Button(ax_record, 'Record current')
        self.btn_record.on_clicked(self.on_record)

        ax_sweep = self.fig.add_axes([0.17, 0.18, 0.1, 0.04])
        self.btn_sweep = Button(ax_sweep, 'Record sweep')
        self.btn_sweep.on_clicked(self.on_sweep)

        ax_clear = self.fig.add_axes([0.29, 0.18, 0.1, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear sinogram')
        self.btn_clear.on_clicked(self.on_clear)

        ax_recon = self.fig.add_axes([0.41, 0.18, 0.1, 0.04])
        self.btn_recon = Button(ax_recon, 'Reconstruct')
        self.btn_recon.on_clicked(self.on_reconstruct)

        ax_load = self.fig.add_axes([0.53, 0.18, 0.1, 0.04])
        self.btn_load = Button(ax_load, 'Load image')
        self.btn_load.on_clicked(self.on_load)

        ax_save = self.fig.add_axes([0.65, 0.18, 0.1, 0.04])
        self.btn_save = Button(ax_save, 'Save sinogram')
        self.btn_save.on_clicked(self.on_save)

        ax_loadsino = self.fig.add_axes([0.77, 0.18, 0.1, 0.04])
        self.btn_loadsino = Button(ax_loadsino, 'Load sinogram')
        self.btn_loadsino.on_clicked(self.on_load_sino)

        ax_export = self.fig.add_axes([0.89, 0.18, 0.1, 0.04])
        self.btn_export = Button(ax_export, 'Export PNGs')
        self.btn_export.on_clicked(self.on_export)

        # Radio buttons and checkboxes
        ax_filter = self.fig.add_axes([0.05, 0.05, 0.1, 0.12])
        self.rad_filter = RadioButtons(ax_filter, ['ramp', 'shepp-logan', 'hann', 'hamming', 'cosine', 'none'])
        self.rad_filter.on_clicked(self.on_filter)

        ax_interp = self.fig.add_axes([0.17, 0.05, 0.1, 0.07])
        self.rad_interp = RadioButtons(ax_interp, ['linear', 'nearest'])
        self.rad_interp.on_clicked(self.on_interp)

        ax_intensity = self.fig.add_axes([0.29, 0.05, 0.1, 0.05])
        self.chk_intensity = CheckButtons(ax_intensity, ['Show intensity'], [self.show_intensity])
        self.chk_intensity.on_clicked(self.on_intensity)

        ax_noise = self.fig.add_axes([0.41, 0.05, 0.1, 0.12])
        self.rad_noise = RadioButtons(ax_noise, ['none', 'low', 'medium', 'high'])
        self.rad_noise.on_clicked(self.on_noise)

        # Text boxes for numeric parameters
        ax_det = self.fig.add_axes([0.53, 0.08, 0.07, 0.04])
        self.txt_det = TextBox(ax_det, 'Detectors', initial=str(self.n_detectors))
        self.txt_det.on_submit(self.on_detectors)

        ax_out = self.fig.add_axes([0.63, 0.08, 0.07, 0.04])
        self.txt_out = TextBox(ax_out, 'Recon size', initial=str(self.output_size))
        self.txt_out.on_submit(self.on_output_size)

        ax_step = self.fig.add_axes([0.73, 0.08, 0.07, 0.04])
        self.txt_step = TextBox(ax_step, 'Sweep step', initial=str(self.sweep_step))
        self.txt_step.on_submit(self.on_step)

        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.timer_tick)

        self.metric_text = self.fig.text(0.85, 0.05, '', transform=self.fig.transFigure)

    # ------------------------------ Widget callbacks -----------------------
    def on_slider(self, val):
        self.theta = val
        self._update_all()

    def on_play(self, event):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.label.set_text('Pause')
            self.timer.start()
        else:
            self.btn_play.label.set_text('Play')
            self.timer.stop()

    def timer_tick(self):
        new_val = self.slider.val + 0.5
        if new_val > 180:
            new_val -= 180
        self.slider.set_val(new_val)

    def on_record(self, event):
        self.sino, self.theta_rec = append_measurement(self.sino, self.theta_rec, self.current_L, self.theta)
        self._update_sino()

    def on_sweep(self, event):
        start = self.theta
        angles = np.arange(start, 180.0, self.sweep_step)
        for i, ang in enumerate(angles):
            L = compute_projection(self.img, ang, self.n_detectors)
            self.sino, self.theta_rec = append_measurement(self.sino, self.theta_rec, L, ang)
            if i % 5 == 0:
                self.slider.set_val(ang)
                plt.pause(0.001)
        self._update_all()

    def on_clear(self, event):
        self.sino = np.zeros((self.n_detectors, 0), dtype=np.float32)
        self.theta_rec = []
        self.recon = None
        self.metric_text.set_text('')
        self._update_sino()

    def on_reconstruct(self, event):
        self.recon = reconstruct(self.sino, self.theta_rec, self.filter, self.interp, self.output_size)
        if self.recon is not None:
            self.img_im.set_data(self.recon)
            if self.img.shape == self.recon.shape:
                p = psnr(self.img, self.recon, data_range=1)
                s = ssim(self.img, self.recon, data_range=1)
                self.metric_text.set_text(f'PSNR {p:.2f} dB\nSSIM {s:.3f}')
        self.fig.canvas.draw_idle()

    def on_load(self, event):
        try:
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            path = filedialog.askopenfilename(filetypes=[('Image', '*.png;*.jpg;*.jpeg')])
            root.destroy()
            if path:
                self.img = load_image(path, size=self.size)
                self.img *= self.mask
                self.img_im.set_data(self.img)
                self.on_clear(None)
                self._update_all()
        except Exception as e:
            print('Load failed:', e)

    def on_save(self, event):
        save_npz('sinogram.npz', self.sino, self.theta_rec)

    def on_load_sino(self, event):
        if os.path.exists('sinogram.npz'):
            self.sino, self.theta_rec = load_npz('sinogram.npz')
            self._update_all()

    def on_export(self, event):
        export_pngs('ct_demo', self.img, self.sino, self.recon)

    def on_filter(self, label):
        self.filter = label

    def on_interp(self, label):
        self.interp = label

    def on_intensity(self, label):
        self.show_intensity = not self.show_intensity
        self._update_projection()

    def on_noise(self, label):
        mapping = {'none':0.0, 'low':0.01, 'medium':0.05, 'high':0.1}
        self.noise_sigma = mapping[label]
        self._update_projection()

    def on_detectors(self, text):
        try:
            val = int(text)
            if val > 8:
                self.n_detectors = val
                self.on_clear(None)
                self.current_L = compute_projection(self.img, self.theta, self.n_detectors)
                self.proj_line.set_xdata(np.arange(self.n_detectors))
                self._update_all()
        except ValueError:
            pass

    def on_output_size(self, text):
        try:
            val = int(text)
            if val > 0:
                self.output_size = val
        except ValueError:
            pass

    def on_step(self, text):
        try:
            val = float(text)
            if val > 0:
                self.sweep_step = val
        except ValueError:
            pass

    # --------------------------- Update functions -------------------------
    def _update_all(self):
        self._update_projection()
        self._update_sino()
        self._update_geometry()

    def _update_projection(self):
        self.current_L = compute_projection(self.img, self.theta, self.n_detectors)
        if self.show_intensity:
            y = np.exp(-self.current_L)
            if self.noise_sigma > 0:
                y = np.clip(y + np.random.normal(0, self.noise_sigma, size=y.shape), 0, 1)
        else:
            y = self.current_L
        self.proj_line.set_data(np.arange(len(y)), y)
        self.ax_proj.set_title(f'Projection at θ = {self.theta:.1f}°')
        self.ax_proj.relim()
        self.ax_proj.autoscale_view()
        self.fig.canvas.draw_idle()

    def _update_sino(self):
        if self.sino.size == 0:
            data = np.zeros((self.n_detectors, 1))
        else:
            data = self.sino
        self.sino_im.set_data(data)
        self.sino_im.set_extent([0, data.shape[1], 0, self.n_detectors])
        self.ax_sino.set_xlim(0, max(1, data.shape[1]))
        self.ax_sino.set_ylim(self.n_detectors, 0)
        self.fig.canvas.draw_idle()

    def _update_geometry(self):
        rad = np.deg2rad(self.theta)
        cx = cy = self.size / 2
        length = self.size
        x0 = cx - length * np.sin(rad) / 2
        y0 = cy + length * np.cos(rad) / 2
        x1 = cx + length * np.sin(rad) / 2
        y1 = cy - length * np.cos(rad) / 2
        self.det_line.set_data([x0, x1], [y0, y1])
        offs = np.linspace(-length/4, length/4, 3)
        for off, line in zip(offs, self.ray_lines):
            rx0 = cx + off * np.cos(rad) - length/2 * np.sin(rad)
            ry0 = cy + off * np.sin(rad) + length/2 * np.cos(rad)
            rx1 = cx + off * np.cos(rad) + length/2 * np.sin(rad)
            ry1 = cy + off * np.sin(rad) - length/2 * np.cos(rad)
            line.set_data([rx0, rx1], [ry0, ry1])
        self.fig.canvas.draw_idle()

# ------------------------------------ main -----------------------------------

def main():
    if 'scikit-image' not in sys.modules:
        try:
            import skimage
        except ImportError:
            print('scikit-image is required. Install with "pip install scikit-image"')
            return
    CTDemo()
    plt.show()


if __name__ == '__main__':
    main()
