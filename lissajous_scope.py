#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lissajous / Sawtooth Sweep Oscilloscope-style Animator (v4.6 UI compact + directional guides)
--------------------------------------------------------------------------------------------
Updates (per request):
1) XY補助線を「x(t) と y(t) のグラフがある方向のみに伸びる半クロスヘア」に変更。
   - XY 内の水平ガイド: 現在点から“右方向”のみに伸びる（y(t) が右にあるため）。
   - XY 内の垂直ガイド: 現在点から“下方向”のみに伸びる（x(t) が下にあるため）。
   * いずれも XY 軸内にクリップ。
2) Controls ボックスのレイアウトを再調整。
   - すべての UI がパネル内に収まるよう縦間隔/サイズを縮小し、重なりを解消。
   - テキストボックスは固定幅、ラジオボタンの枠をコンパクト化。
3) 既存のガイド ON/OFF も維持（チェックボックス）。

Base: v4.5 UI compact
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, RadioButtons, Button, TextBox, CheckButtons
from matplotlib.lines import Line2D

# ---- Render speed tweaks ----
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 0.2
matplotlib.rcParams['agg.path.chunksize'] = 20000

# ---- Wave utilities ----

def sawtooth(theta):
    x = theta / (2.0 * np.pi)
    return 2.0 * (x - np.floor(x + 0.5))

def wave(A, omega, phi, t, kind='sin'):
    arg = omega * t + phi
    return A * np.sin(arg) if kind == 'sin' else A * sawtooth(arg)

def rational_approx(r, max_den=64, tol=1e-6):
    if r <= 0:
        return (0, 1)
    p_best, q_best = 1, 1
    err_best = abs(r - 1.0)
    for q in range(1, max_den+1):
        p = int(round(r * q))
        if p == 0:
            continue
        err = abs(r - p/q)
        if err < err_best - 1e-12:
            err_best = err
            p_best, q_best = p, q
            if err_best < tol:
                break
    return (p_best, q_best)

def insert_nans_for_jumps_arr(x, y, thr_x, thr_y):
    if x.size < 2:
        return x, y
    dx = np.abs(np.diff(x))
    dy = np.abs(np.diff(y))
    cut = (dx > thr_x) | (dy > thr_y)
    if not np.any(cut):
        return x, y
    xn = []
    yn = []
    for i in range(len(x)-1):
        xn.append(x[i]); yn.append(y[i])
        if cut[i]:
            xn.append(np.nan); yn.append(np.nan)
    xn.append(x[-1]); yn.append(y[-1])
    return np.array(xn), np.array(yn)

def decimate_xy(x, y, max_points):
    n = len(x)
    if n <= max_points:
        return np.asarray(x), np.asarray(y)
    step = max(1, n // max_points)
    return np.asarray(x[::step]), np.asarray(y[::step])

# ---- Parameters ----
Ax0, wx0, phix0 = 1.0, 3.0, 0.0
Ay0, wy0, phiy0 = 1.0, 2.0, 0.0
FPS = 60
DT = 1.0 / FPS
WIN = 4.0

MAX_POINTS_KEEP = 1_000_000
MAX_DRAW_XY     = 12_000
MAX_DRAW_TIME   = 3_000
MAX_FULL_POINTS = 20_000

MARGIN = 1.1

# ---- Figure & layout ----
plt.rcParams['figure.figsize'] = (14.0, 8.0)
fig = plt.figure(constrained_layout=False)
# Left area for plots, right column for a single control box
gs = fig.add_gridspec(2, 3,
                      width_ratios=[4.0, 1.6, 2.2],
                      height_ratios=[4.0, 1.9],
                      wspace=0.35, hspace=0.30)

ax_xy = fig.add_subplot(gs[0, 0])
ax_xt = fig.add_subplot(gs[1, 0], sharex=ax_xy)
ax_yt = fig.add_subplot(gs[0, 1])

# A single control box (panel) occupying the entire right column
ax_panel = fig.add_subplot(gs[:, 2])
ax_panel.set_title("Controls", loc='left', fontsize=11, pad=4)
ax_panel.set_xticks([]); ax_panel.set_yticks([])
ax_panel.set_frame_on(True)
for spine in ax_panel.spines.values():
    spine.set_alpha(0.25)
ax_panel.set_facecolor((0.97, 0.97, 0.97))

ax_xy.set_title("Lissajous / Sweep (gray: full, red: trace)")
ax_xt.set_title("X input x(t) — time axis downward (+), X matches XY")
ax_yt.set_title("Y input y(t) — at right of XY")

ax_xy.set_aspect('equal', adjustable='box')
ax_xy.grid(True, alpha=0.2)

ax_xt.set_ylim(WIN, 0)
ax_xt.set_ylabel("time (s, + downward)")
ax_xt.set_xlabel("x amplitude")

ax_yt.set_xlim(0, WIN)
ax_yt.set_xlabel("time (s)")
ax_yt.set_ylabel("y amplitude")

# ---- Plot artists ----
xy_full_line, = ax_xy.plot([], [], lw=1.0, alpha=0.85)
xy_traj_line, = ax_xy.plot([], [], lw=2.0, color='red')
xt_line, = ax_xt.plot([], [], lw=1.3)
yt_line, = ax_yt.plot([], [], lw=1.3)

xy_dot, = ax_xy.plot([], [], marker='o', ms=8, mfc='white', mec='red', mew=1.5, linestyle='None')
xt_dot, = ax_xt.plot([], [], marker='o', ms=7, mfc='white', mec='C0', mew=1.3, linestyle='None')
yt_dot, = ax_yt.plot([], [], marker='o', ms=7, mfc='white', mec='C1', mew=1.3, linestyle='None')

# ---- Directional guide lines inside XY (half crosshair) ----
# We use Line2D segments so we can control the extent per frame
xy_hseg = Line2D([], [], linestyle='--', lw=1.0, alpha=0.55, color='0.3')  # from point to RIGHT only
xy_vseg = Line2D([], [], linestyle='--', lw=1.0, alpha=0.55, color='0.3')  # from bottom to point only
ax_xy.add_line(xy_hseg)
ax_xy.add_line(xy_vseg)

# x(t): vertical line at x_now (full height, but clipped to the axes region)
xt_vline = ax_xt.axvline(0.0, linestyle='--', lw=1.0, alpha=0.45, color='0.3', visible=True)
# y(t): horizontal line at y_now (full width, but clipped to the axes region)
yt_hline = ax_yt.axhline(0.0, linestyle='--', lw=1.0, alpha=0.45, color='0.3', visible=True)

for art in (xy_full_line, xy_traj_line, xt_line, yt_line,
            xy_dot, xt_dot, yt_dot,
            xy_hseg, xy_vseg, xt_vline, yt_hline):
    art.set_animated(True)

# ---- Controls (single compact panel) ----
from matplotlib.transforms import Bbox

def add_in_panel(x0, y0, w, h):
    bb = ax_panel.get_position()
    return fig.add_axes([bb.x0 + x0 * bb.width,
                         bb.y0 + y0 * bb.height,
                         w * bb.width, h * bb.height])

# Top: waveform radios for X, Y (compact)
raxX = add_in_panel(0.05, 0.73, 0.43, 0.20)
raxY = add_in_panel(0.52, 0.73, 0.43, 0.20)
rX = RadioButtons(raxX, ('sin', 'saw'), active=0)
rY = RadioButtons(raxY, ('sin', 'saw'), active=0)
raxX.set_title("X waveform", pad=2, fontsize=9)
raxY.set_title("Y waveform", pad=2, fontsize=9)

for axr in (raxX, raxY):
    for spine in axr.spines.values():
        spine.set_alpha(0.2)

# Middle: sliders + text boxes (tighter spacing)
slider_specs = [
    ('Ax',       0.0, 5.0,    Ax0),
    ('omega_x',  0.1, 20.0,   wx0),
    ('phi_x',    0.0, 2*np.pi,phix0),
    ('Ay',       0.0, 5.0,    Ay0),
    ('omega_y',  0.1, 20.0,   wy0),
    ('phi_y',    0.0, 2*np.pi,phiy0),
    ('speed',    0.1, 5.0,    1.0),
]

sliders = {}
textboxes = {}

y_base = 0.37
row_h  = 0.052  # tighter
for i, (label, vmin, vmax, vinit) in enumerate(slider_specs):
    y = y_base + (len(slider_specs)-1 - i) * row_h
    ax_s = add_in_panel(0.25, y, 0.35, 0.042)
    s = Slider(ax_s, label, vmin, vmax, valinit=vinit, valstep=0.001)
    sliders[label] = s
    ax_t = add_in_panel(0.78, y, 0.17, 0.042)
    tb = TextBox(ax_t, '', initial=f'{vinit:.6g}')
    textboxes[label] = tb

# Buttons row (no overlap)
ax_bpause = add_in_panel(0.05, 0.27, 0.42, 0.062); b_pause = Button(ax_bpause, 'Pause/Play')
ax_breset = add_in_panel(0.53, 0.27, 0.42, 0.062); b_reset = Button(ax_breset, 'Reset')

# Guides checkbox (clearly below buttons)
ax_chk = add_in_panel(0.05, 0.16, 0.62, 0.085)
chk = CheckButtons(ax_chk, labels=['guides'], actives=[True])

# ---- State ----
t_hist = []
x_hist = []
y_hist = []
t_now = 0.0
running = True

x_full = np.array([]); y_full = np.array([])
last_x = None; last_y = None

# ---- Callbacks/util ----

def current_params():
    Ax = sliders['Ax'].val; wx = sliders['omega_x'].val; phx = sliders['phi_x'].val; kindx = rX.value_selected
    Ay = sliders['Ay'].val; wy = sliders['omega_y'].val; phy = sliders['phi_y'].val; kindy = rY.value_selected
    speed = sliders['speed'].val
    return Ax, wx, phx, kindx, Ay, wy, phy, kindy, speed


def recompute_full_curve():
    global x_full, y_full
    Ax, wx, phx, kindx, Ay, wy, phy, kindy, _ = current_params()
    ratio = wx / max(wy, 1e-12)
    p, q = rational_approx(ratio, max_den=64, tol=1e-7)
    T_full = 2.0 * np.pi * q / max(wy, 1e-9)

    Nbase = min(4000 * q, 120_000)
    t = np.linspace(0.0, T_full, int(Nbase))
    xf = wave(Ax, wx, phx, t, kindx)
    yf = wave(Ay, wy, phy, t, kindy)

    thr_x = 1.5 * max(Ax, 1e-12)
    thr_y = 1.5 * max(Ay, 1.0e-12)
    xf, yf = insert_nans_for_jumps_arr(xf, yf, thr_x, thr_y)

    xf, yf = decimate_xy(xf, yf, MAX_FULL_POINTS)

    x_full, y_full = xf, yf
    xy_full_line.set_data(x_full, y_full)

    R = MARGIN * max(Ax, Ay, 1.0)
    ax_xy.set_xlim(-R, R); ax_xy.set_ylim(-R, R); ax_xy.set_aspect('equal', adjustable='box')

    ax_xt.set_ylim(WIN, 0)
    ax_yt.set_xlim(0, WIN)
    ax_yt.set_ylim(-MARGIN*max(Ay, 1.0), MARGIN*max(Ay, 1.0))

    fig.canvas.draw_idle()


def reset_hist():
    global t_hist, x_hist, y_hist, t_now, last_x, last_y
    t_hist = []; x_hist = []; y_hist = []; t_now = 0.0; last_x = None; last_y = None
    xy_traj_line.set_data([], []); xt_line.set_data([], []); yt_line.set_data([], [])
    xy_dot.set_data([], []); xt_dot.set_data([], []); yt_dot.set_data([], [])
    # reset guides visibility
    vis = chk.get_status()[0]
    for ln in (xy_hseg, xy_vseg, xt_vline, yt_hline):
        ln.set_visible(vis)


def on_pause_clicked(event):
    global running; running = not running

def on_reset_clicked(event):
    reset_hist()

b_pause.on_clicked(on_pause_clicked)
b_reset.on_clicked(on_reset_clicked)


def sync_textbox_from_slider(label):
    tb = textboxes[label]; sl = sliders[label]; tb.set_val(f'{sl.val:.6g}')


def on_any_param_changed(val):
    reset_hist()
    for label in sliders: sync_textbox_from_slider(label)
    recompute_full_curve()

for s in sliders.values():
    s.on_changed(on_any_param_changed)
rX.on_clicked(on_any_param_changed); rY.on_clicked(on_any_param_changed)


def make_textbox_handler(label):
    def _on_submit(text):
        try:
            v = float(text)
        except ValueError:
            sync_textbox_from_slider(label); return
        sl = sliders[label]
        v = max(sl.valmin, min(sl.valmax, v))
        if abs(v - sl.val) > 1e-12:
            sl.set_val(v)
        else:
            reset_hist(); sync_textbox_from_slider(label); recompute_full_curve()
    return _on_submit

for label, tb in textboxes.items():
    tb.on_submit(make_textbox_handler(label))

# Guides ON/OFF

def on_toggle(label):
    vis = chk.get_status()[0]
    for ln in (xy_hseg, xy_vseg, xt_vline, yt_hline):
        ln.set_visible(vis)
    fig.canvas.draw_idle()

chk.on_clicked(on_toggle)

# Match x(t) width to XY

def match_xt_width(event=None):
    bb_xy = ax_xy.get_position()
    bb_xt = ax_xt.get_position()
    ax_xt.set_position([bb_xy.x0, bb_xt.y0, bb_xy.width, bb_xt.height])

cid = fig.canvas.mpl_connect('draw_event', match_xt_width)

# Initial
recompute_full_curve(); match_xt_width()

# ---- Animation ----


def init_anim():
    xy_full_line.set_data(x_full, y_full)
    xy_traj_line.set_data([], [])
    xt_line.set_data([], [])
    yt_line.set_data([], [])
    xy_dot.set_data([], []); xt_dot.set_data([], []); yt_dot.set_data([], [])
    vis = chk.get_status()[0]
    for ln in (xy_hseg, xy_vseg, xt_vline, yt_hline):
        ln.set_visible(vis)
    return (xy_full_line, xy_traj_line, xt_line, yt_line,
            xy_dot, xt_dot, yt_dot, xy_hseg, xy_vseg, xt_vline, yt_hline)


def update(frame):
    global t_now, last_x, last_y

    if not running:
        return (xy_full_line, xy_traj_line, xt_line, yt_line,
                xy_dot, xt_dot, yt_dot, xy_hseg, xy_vseg, xt_vline, yt_hline)

    Ax, wx, phx, kindx, Ay, wy, phy, kindy, speed = current_params()
    t_now += DT * speed

    x_now = wave(Ax, wx, phx, t_now, kindx)
    y_now = wave(Ay, wy, phy, t_now, kindy)

    thr_x = 1.5 * max(Ax, 1e-12)
    thr_y = 1.5 * max(Ay, 1e-12)
    if last_x is not None and (abs(x_now - last_x) > thr_x or abs(y_now - last_y) > thr_y):
        x_hist.append(float('nan')); y_hist.append(float('nan')); t_hist.append(t_now)
    last_x, last_y = x_now, y_now

    x_hist.append(x_now); y_hist.append(y_now); t_hist.append(t_now)

    tmin = max(0.0, t_now - WIN)
    n = len(t_hist)
    i0 = 0
    if n > 0 and t_hist[0] < tmin:
        lo, hi = 0, n-1
        while lo < hi:
            mid = (lo + hi) // 2
            if t_hist[mid] < tmin:
                lo = mid + 1
            else:
                hi = mid
        i0 = lo

    t_win = t_hist[i0:]
    x_win = x_hist[i0:]
    y_win = y_hist[i0:]

    L = len(t_win)
    stride_t = max(1, L // MAX_DRAW_TIME)
    t_rel_d = [(tv - tmin) for tv in t_win[::stride_t]]
    x_d = x_win[::stride_t]
    y_d = y_win[::stride_t]

    yt_line.set_data(t_rel_d, y_d)
    xt_line.set_data(x_d, t_rel_d)
    ax_xt.set_ylim(WIN, 0)

    t_rel_now = t_now - tmin
    xy_dot.set_data([x_now], [y_now])
    xt_dot.set_data([x_now], [t_rel_now])
    yt_dot.set_data([t_rel_now], [y_now])

    # Update directional guides in XY (half crosshair)
    if xy_hseg.get_visible() or xy_vseg.get_visible():
        xmin, xmax = ax_xy.get_xlim()
        ymin, ymax = ax_xy.get_ylim()
        # horizontal: from point to RIGHT only
        xy_hseg.set_data([x_now, xmax], [y_now, y_now])
        # vertical: from BOTTOM to point only
        xy_vseg.set_data([x_now, x_now], [ymin, y_now])

        # x(t) / y(t) guides
        xt_vline.set_xdata([x_now, x_now])
        yt_hline.set_ydata([y_now, y_now])

    x_xy_d, y_xy_d = decimate_xy(x_hist, y_hist, MAX_DRAW_XY)
    xy_traj_line.set_data(x_xy_d, y_xy_d)

    if len(t_hist) > MAX_POINTS_KEEP:
        drop = len(t_hist) - MAX_POINTS_KEEP
        del t_hist[:drop]; del x_hist[:drop]; del y_hist[:drop]

    return (xy_full_line, xy_traj_line, xt_line, yt_line,
            xy_dot, xt_dot, yt_dot, xy_hseg, xy_vseg, xt_vline, yt_hline)

ani = FuncAnimation(fig, update, init_func=init_anim, interval=1000.0*DT, blit=True)

plt.show()

if __name__ == "__main__":
    pass
