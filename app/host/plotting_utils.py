import math
import os
from math import pi
from packaging.version import Version

import numpy as np
import pandas as pd
import prospect.io.read_results as reader
from astropy.coordinates import SkyCoord
from astropy.cosmology import WMAP9 as cosmo
from astropy.io import fits
from astropy.visualization import AsinhStretch
from astropy.visualization import PercentileInterval
from astropy.wcs import WCS
from bokeh.embed import components
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import Label
from bokeh.models import LabelSet
from bokeh.models import PointDrawTool
from bokeh.models import Range1d
from bokeh.palettes import Category20
# from bokeh.plotting import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import cumsum
from host.models import Filter
from host.photometric_calibration import maggies_to_mJy
from host.prospector import build_obs
from host.models import SEDFittingResult
from host.models import HostSpectrum
from bokeh.models import CustomJS
from host.object_store import ObjectStore
from django.conf import settings
from bokeh.io import curdoc

from host.host_utils import get_local_aperture_size

from host.log import get_logger
logger = get_logger(__name__)

EDITABLE_APERTURES_HANDLE_JS = """
const ellipses = ellipse_source.data;
const handles = handle_source.data;
const state = callback_state.data;

// Updating handle_source ourselves causes its change callback to run again.
// This guard prevents recursive callback execution.
if (state.updating[0]) {
    return;
}

state.updating[0] = true;

try {
    // Each ellipse n owns exactly two handles, laid out at fixed
    // positions 2n (major) and 2n+1 (minor) in the shared handle source.
    for (let n = 0; n < ellipses.x.length; n++) {
        // const majorIdx = 2 * n;
        // const minorIdx = 2 * n + 1;

        const majorIdx = major_handle_indices[n];
        const minorIdx = minor_handle_indices[n];

        const cx = ellipses.x[n];
        const cy = ellipses.y[n];

        const majorDx = handles.x[majorIdx] - cx;
        const majorDy = handles.y[majorIdx] - cy;

        // Prevent a zero-sized major axis.
        const semiMajor = Math.max(
            minimum_radius,
            Math.hypot(majorDx, majorDy),
        );

        // Bokeh ellipse angles are measured in radians.
        const theta = Math.atan2(majorDy, majorDx);
        
        let semiMinor = semiMajor;

        // if the aperture handle for minor axis exists
        if (minorIdx !== null && minorIdx !== undefined) {
            const minorUnitX = -Math.sin(theta);
            const minorUnitY = Math.cos(theta);
    
            const minorDx = handles.x[minorIdx] - cx;
            const minorDy = handles.y[minorIdx] - cy;
    
            // Project the dragged minor handle onto the true minor axis so it can't drift away from a perpendicular position.
            const projectedMinorRadius = Math.abs(
                minorDx * minorUnitX +
                minorDy * minorUnitY
            );

            // Keep the aperture mathematically consistent: semi-major >= semi-minor.
            semiMinor = Math.min(
                semiMajor,
                Math.max(minimum_radius, projectedMinorRadius),
            );

            // Snap control back onto axes
            handles.x[minorIdx] = cx + semiMinor * minorUnitX;
            handles.y[minorIdx] = cy + semiMinor * minorUnitY;
    
        }

        ellipses.width[n] = 2.0 * semiMajor;
        ellipses.height[n] = 2.0 * semiMinor;
        ellipses.angle[n] = theta;

        // Snap both controls back onto their exact axes.
        handles.x[majorIdx] = cx + semiMajor * Math.cos(theta);
        handles.y[majorIdx] = cy + semiMajor * Math.sin(theta);

        // Publish the latest unsaved geometry for this ellipse to the
        // surrounding page. Fired for every ellipse on every callback run;
        // the page-level listener just overwrites its cache entry, so this
        // is cheap and avoids tracking which specific point moved.
        window.dispatchEvent(
            new CustomEvent("blast:aperture-change", {
                detail: {
                    apertureId: aperture_ids[n],
                    apertureType: aperture_types[n],
                    x: cx,
                    y: cy,
                    semiMajor: semiMajor,
                    semiMinor: semiMinor,
                    thetaRadians: theta,
                },
            })
        );
    }

    ellipse_source.change.emit();
    handle_source.change.emit();
} finally {
    state.updating[0] = false;
    callback_state.change.emit();
}
"""

EDITABLE_APERTURE_CENTER_JS = """
const center = center_source.data;
const prev = center_prev_source.data;
const ellipses = ellipse_source.data;
const handles = handle_source.data;
const state = callback_state.data;

// share guard with EDITABLE_APERTURES_HANDLE_JS. 
// this writes to ellipse_source/handle_source,
// rotation must not recompute geometry on top of it

if (state.updating[0]) {
return;
}

state.updating[0] = true;

try {
    for (let n = 0; n < center.x.length; n++) {
        const dx = center.x[n] - prev.x[n];
        const dy = center.y[n] - prev.y[n];

        if (dx === 0. && dy === 0) {
            continue;
        }

        // move ellipse by distance "grab region" was moved

        ellipses.x[n] += dx;
        ellipses.y[n] += dy;

        // move both handles by same delta 
        const majorIdx = major_handle_indices[n];
        const minorIdx = minor_handle_indices[n];
        handles.x[majorIdx] += dx;
        handles.y[majorIdx] += dy;
        if (minorIdx !== null && minorIdx !== undefined) {
            handles.x[minorIdx] += dx;
            handles.y[minorIdx] += dy;
        }

        prev.x[n] = center.x[n];
        prev.y[n] = center.y[n];

        window.dispatchEvent(
            new CustomEvent("blast:aperture-translate", {
                detail: {
                    apertureId: aperture_ids[n],
                    apertureType: aperture_types[n],
                    x: ellipses.x[n],
                    y: ellipses.y[n],
                    semiMajor: ellipses.width[n] / 2.0,
                    semiMinor: ellipses.height[n] / 2.0,
                    thetaRadians: ellipses.angle[n],                    
                },
            })    
        );

    }

    ellipse_source.change.emit();
    handle_source.change.emit();

} finally {
    state.updating[0] = false;
    callback_state.change.emit();
}

"""

APERTURE_LIVE_READOUT_JS = """
// recompute readout from current ellips egeometry. 

// Registered on ellipse_source, which is single source that both the handle callback and center callback  write to

// `cd` is WCS CD matrix flattened in arsec/pixel

for (let n = 0; n < ellipses.x.length; n++) {
    const semiMajor = ellipses.width[n] / 2.0;
    const semiMinor = ellipses.height[n] / 2.0;
    const theta = ellipses.angle[n];

    // Axis end-point offsets from center in pixels
    const majDx = semiMajor * Math.cos(theta);
    const majDy = semiMajor * Math.sin(theta);
    const minDx = -semiMinor * Math.sin(theta);
    const minDy = semiMinor * Math.cos(theta);

    // Pixels -> arcsec
    const majEast  = cd[0] * majDx + cd[1] * majDy;
    const majNorth = cd[2] * majDx + cd[3] * majDy;
    const minEast  = cd[0] * minDx + cd[1] * minDy;
    const minNorth = cd[2] * minDx + cd[3] * minDy;

    const aArcsec = Math.hypot(majEast, majNorth);
    const bArcsec = Math.hypot(minEast, minNorth);

    // position angle of major axis folded to [0, 180) 
    let pa = Math.atan2(majEast, majNorth) * 180.0 / Math.PI;
    pa = ((pa % 180.0) + 180.0) % 180.0;

    let text = aperture_labels[n] + ":  a = " + aArcsec.toFixed(2) + "\\u2033";
    if (arcsec_per_kpc) {
        text += " (" + (aArcsec / arcsec_per_kpc).toFixed(2) + " kpc)";
    }
    text += ",  b = " + bArcsec.toFixed(2) + "\\u2033";
    if (arcsec_per_kpc) {
        text += " (" + (bArcsec / arcsec_per_kpc).toFixed(2) + " kpc)";
    }
    text += ",  PA = " + pa.toFixed(1) + "\\u00B0 E of N";

    readout_labels[n].text = text;

    }
"""

APERTURE_READOUT_POS_JS = """
// bokeh coords measured upward from bottom of plot frame, 

// pin to top left by trackign frame height, only known client-side after layout

const h = fig.inner_height;
if (h == null || h <= 0) {
    return;
}
for (let n = 0; n < readout_labels.length; n++) {
    readout_labels[n].y = h - top_margin - n * line_height;
}

"""


def scale_image(image_data):
    transform = AsinhStretch() + PercentileInterval(99.5)
    scaled_data = transform(image_data)

    return scaled_data


def plot_image(image_data, figure):
    # sometimes low image mins mess up the plotting

    perc01 = np.nanpercentile(image_data, 1)

    image_data = np.nan_to_num(image_data, nan=perc01)
    image_data = image_data + abs(np.amin(image_data)) + 0.1

    scaled_image = scale_image(image_data)

    figure.image(image=[scaled_image])
    figure.image(
        image=[scaled_image],
        x=0,
        y=0,
        dw=np.shape(image_data)[1],
        dh=np.shape(image_data)[0],
        level="image",
    )


def plot_position(object, wcs, plotting_kwargs=None, plotting_func=None):
    """
    Plot position of object on a cutout.
    """
    obj_ra, obj_dec = object.ra_deg, object.dec_deg
    sky_position = SkyCoord(ra=obj_ra, dec=obj_dec, unit="deg")
    x_pixel, y_pixel = wcs.world_to_pixel(sky_position)
    plotting_func([x_pixel], [y_pixel], **plotting_kwargs)
    return None


def plot_aperture(figure, aperture, wcs, plotting_kwargs=None):
    aperture = aperture.to_pixel(wcs)
    theta_rad = aperture.theta.rad
    x, y = aperture.positions
    plot_dict = {
        "x": x,
        "y": y,
        "width": aperture.a * 2,
        "height": aperture.b * 2,
        "angle": theta_rad,
        "fill_color": "#231f25", 
        "fill_alpha": 0.1,
        "line_width": 4,
    }

    plot_dict = {**plot_dict, **plotting_kwargs}
    figure.ellipse(**plot_dict)
    return figure

def plot_editable_apertures(figure, apertures, wcs, minimum_radius=1.0, center_grab_multiplier=0.1, min_center_grab_radius=4.0):
    """
    Draw one or more editable apertures sharing a single PointDrawTool.

    `apertures` is a list of dicts, each with:
        - aperture_id
        - aperture_type ("local" or "global")
        - sky_aperture
        - line_color
        - legend_label
    """
    ellipse_x, ellipse_y = [], []
    ellipse_w, ellipse_h, ellipse_angle = [], [], []
    ellipse_line_color, ellipse_legend = [], []

    GHOST_COLORS = {
        "local": "#56c4ff",   # light blue
        "global": "#b2df8a",  # light green
    }
    ghost_color = []


    handle_x, handle_y, handle_color, handle_axis = [], [], [], []
    aperture_ids, aperture_types = [], []

    grab_x, grab_y, grab_radius = [], [], []

    major_handle_indices, minor_handle_indices = [], []

    for ap in apertures:
        pixel_aperture = ap["sky_aperture"].to_pixel(wcs)
        position = np.asarray(pixel_aperture.positions, dtype=float)

        center_x = float(position[0])
        center_y = float(position[1])
        semi_major = float(pixel_aperture.a)
        semi_minor = float(pixel_aperture.b)
        theta = float(pixel_aperture.theta.value)

        # guard forcing local aperture to render as circular even before edits begin, remove later.
        if ap["aperture_type"] == "local":
            semi_minor = semi_major

        if semi_major <= 0:
            raise ValueError("The aperture semi-major radius must be positive.")
        if semi_minor <= 0:
            raise ValueError("The aperture semi-minor radius must be positive.")

        # Preserve the mathematical meaning of major and minor axes.
        if semi_minor > semi_major:
            semi_major, semi_minor = semi_minor, semi_major
            theta += math.pi / 2.0

        ellipse_x.append(center_x)
        ellipse_y.append(center_y)
        ellipse_w.append(2.0 * semi_major)
        ellipse_h.append(2.0 * semi_minor)
        ellipse_angle.append(theta)
        ellipse_line_color.append(ap["line_color"])
        ellipse_legend.append(ap["legend_label"])
        ghost_color.append(GHOST_COLORS.get(ap["aperture_type"], ""))

        has_minor_handle = ap["aperture_type"] == "global"

        major_handle_x = center_x + semi_major * math.cos(theta)
        major_handle_y = center_y + semi_major * math.sin(theta)

        major_idx = len(handle_x)
        handle_x.append(major_handle_x)
        handle_y.append(major_handle_y)
        handle_color.append(ap["line_color"])
        handle_axis.append("major")
        major_handle_indices.append(major_idx)
 
        if has_minor_handle:
            minor_handle_x = center_x - semi_minor * math.sin(theta)
            minor_handle_y = center_y + semi_minor * math.cos(theta)
 
            minor_idx = len(handle_x)
            handle_x.append(minor_handle_x)
            handle_y.append(minor_handle_y)
            handle_color.append(ap["line_color"])
            handle_axis.append("minor")
            minor_handle_indices.append(minor_idx)
        else:
            minor_handle_indices.append(None)
 
        aperture_ids.append(ap["aperture_id"])
        aperture_types.append(ap["aperture_type"])
 
        grab_x.append(center_x)
        grab_y.append(center_y)
        grab_radius.append(max(min_center_grab_radius, semi_major*center_grab_multiplier))

    ghost_source = ColumnDataSource(
        data={
            "x": ellipse_x,
            "y": ellipse_y,
            "width": ellipse_w,
            "height": ellipse_h,
            "angle": ellipse_angle,
            "line_color":ghost_color,
        }
    )

    ellipse_source = ColumnDataSource(
        data={
            "x": ellipse_x,
            "y": ellipse_y,
            "width": ellipse_w,
            "height": ellipse_h,
            "angle": ellipse_angle,
            "line_color": ellipse_line_color,
            "legend_label": ellipse_legend,
        }
    )

    handle_source = ColumnDataSource(
        data={
            "x": handle_x,
            "y": handle_y,
            "line_color": handle_color,
            "axis": handle_axis,
        }
    )

    center_source = ColumnDataSource(
        data={
            "x": list(grab_x),
            "y": list(grab_y),
            "radius": grab_radius,
        }
    )
    center_prev_source=ColumnDataSource(
        data={
            "x": list(grab_x),
            "y": list(grab_y),
        }
    )

    # Used only to prevent an infinite callback loop when the JavaScript
    # callback snaps the handles back onto their exact axes.
    callback_state = ColumnDataSource(data={"updating": [False]})

    ghost_renderer = figure.ellipse(
        x="x",
        y="y",
        width="width",
        height="height",
        angle="angle",
        source=ghost_source,
        # fill_color="#AAFBAB",
        fill_alpha=0,
        line_color="line_color",
        line_alpha=0.6,
        line_dash="dashed",
        line_width=4,
        name="ghost_aperture_renderer",
    )

    ellipse_renderer = figure.ellipse(
        x="x",
        y="y",
        width="width",
        height="height",
        angle="angle",
        source=ellipse_source,
        # fill_color="#cab2d6",
        fill_alpha=0,
        line_width=4,
        line_color="line_color",
        legend_field="legend_label",
    )

    handle_renderer = figure.scatter(
        x="x",
        y="y",
        source=handle_source,
        marker="circle",
        size=14,
        fill_color="white",
        fill_alpha=1.0,
        line_color="line_color",
        line_width=3,
        selection_fill_color="white",
        selection_line_color="line_color",
        nonselection_fill_color="white",
        nonselection_fill_alpha=1.0,
        nonselection_line_color="line_color",
        nonselection_line_alpha=1.0,
        name="aperture_handle_renderer",
    )

    center_renderer=figure.circle(
        x="x", 
        y="y",
        radius="radius",
        source=center_source,
        fill_alpha=1,
        fill_color= "white",
        line_alpha=1,
        line_color="green",
        line_width=3,
        selection_fill_color="white",
        selection_line_color="green",
        nonselection_fill_color="white",
        nonselection_fill_alpha=1.0,
        nonselection_line_color="green",
        nonselection_line_alpha=1.0,
        name="aperture_center_renderer",
    )

    # A single shared tool drives every editable aperture on this figure.
    handle_tool = PointDrawTool(renderers=[handle_renderer, center_renderer], add=False, name="aperture_handle_tool")
    figure.add_tools(handle_tool)

    handle_callback = CustomJS(
        args={
            "ellipse_source": ellipse_source,
            "handle_source": handle_source,
            "callback_state": callback_state,
            "aperture_ids": aperture_ids,
            "aperture_types": aperture_types,
            "minimum_radius": float(minimum_radius),
            "major_handle_indices" : major_handle_indices,
            "minor_handle_indices" : minor_handle_indices,
        },
        code=EDITABLE_APERTURES_HANDLE_JS,
    )

    # this callback runs repeatedly throughout each drag, should be continuous ???
    handle_source.js_on_change("data", handle_callback)

    center_callback = CustomJS(
        args={
            "center_source":center_source, 
            "center_prev_source":center_prev_source,
            "ellipse_source": ellipse_source,
            "handle_source": handle_source,
            "callback_state": callback_state,
            "aperture_ids": aperture_ids,
            "aperture_types": aperture_types,
            "major_handle_indices" : major_handle_indices,
            "minor_handle_indices" : minor_handle_indices,
        },
        code=EDITABLE_APERTURE_CENTER_JS,
    )

    center_source.js_on_change("data", center_callback)

    return {
        "ellipse_source": ellipse_source,
        "handle_source": handle_source,
        "center_source": center_source,
        "ellipse_renderer": ellipse_renderer,
        "handle_renderer": handle_renderer,
        "handle_tool": handle_tool,
    }

def _pixel_to_arcsec_matrix(wcs):
    """
    Linear pixel -> sky transform in arcsec/pixel, flattened for CustomJS.

    ``wcs.pixel_scale_matrix`` is the CD matrix in deg/pixel taking a pixel
    offset (dx, dy) to intermediate world coordinates (xi, eta), where xi runs
    east (increasing RA * cos(dec)) and eta runs north. It carries the pixel
    scale, any rotation and any flip, so the readout stays correct for cutouts
    that are not north-up/east-left.
    """
    cd = np.asarray(wcs.celestial.pixel_scale_matrix, dtype=float) * 3600.0
    return [float(cd[0, 0]), float(cd[0, 1]), float(cd[1, 0]), float(cd[1, 1])]


def _arcsec_per_kpc(redshift):
    """
    Angular size of 1 proper kpc at ``redshift``, in arcsec, or None when the
    redshift is unusable. This is exactly get_local_aperture_size() with
    apr_kpc=1, and inverting it is what turns an arcsec readout into kpc.
    """
    if redshift is None:
        return None
    try:
        redshift = float(redshift)
    except (TypeError, ValueError):
        return None
    if redshift <= 0:
        return None
    try:
        return float(get_local_aperture_size(redshift, apr_kpc=1.0))
    except Exception as err:
        logger.warning(f"Could not compute angular scale at z={redshift}: {err}")
        return None


def _pixel_geometry(sky_aperture, wcs):
    """(semi_major_px, semi_minor_px, theta_rad), normalised so a >= b."""
    pixel_aperture = sky_aperture.to_pixel(wcs)
    semi_major = float(pixel_aperture.a)
    semi_minor = float(pixel_aperture.b)
    theta = float(pixel_aperture.theta.value)
    if semi_minor > semi_major:
        semi_major, semi_minor = semi_minor, semi_major
        theta += math.pi / 2.0
    return semi_major, semi_minor, theta


def _aperture_readout_text(label, semi_major, semi_minor, theta, cd, arcsec_per_kpc):
    """Python mirror of APERTURE_READOUT_JS, used for the initial label text."""
    matrix = np.array([[cd[0], cd[1]], [cd[2], cd[3]]], dtype=float)
    major = matrix @ np.array([semi_major * np.cos(theta), semi_major * np.sin(theta)])
    minor = matrix @ np.array([-semi_minor * np.sin(theta), semi_minor * np.cos(theta)])

    a_arcsec = float(np.hypot(*major))
    b_arcsec = float(np.hypot(*minor))
    pa = float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)

    text = f"{label}:  a = {a_arcsec:.2f}\u2033"
    if arcsec_per_kpc:
        text += f" ({a_arcsec / arcsec_per_kpc:.2f} kpc)"
    text += f",  b = {b_arcsec:.2f}\u2033"
    if arcsec_per_kpc:
        text += f" ({b_arcsec / arcsec_per_kpc:.2f} kpc)"
    text += f",  PA = {pa:.1f}\u00B0 E of N"
    return text


def add_aperture_readout(
    fig,
    wcs,
    redshift,
    labels,
    colors,
    ellipse_source=None,
    static_geometry=None,
    top_margin=10.0,
    line_height=17.0,
):
    """
    Pin one line of aperture geometry per aperture to the top-left of ``fig``.

    Each line reports the semi-major and semi-minor axes converted from plot
    (pixel) coordinates into arcsec on the sky via the WCS CD matrix, the same
    quantities in kpc at ``redshift``, and the major-axis position angle east
    of north.

    Pass ``ellipse_source`` (the source returned by plot_editable_apertures) to
    make the readout track drags live; pass ``static_geometry`` — a list of
    (semi_major_px, semi_minor_px, theta_rad) — for the non-editable plot.

    One Label per aperture rather than one multi-line Label: Label text wrapping
    is version-dependent, separate Labels are not, and it lets each line take
    its aperture's colour.
    """
    cd = _pixel_to_arcsec_matrix(wcs)
    arcsec_per_kpc = _arcsec_per_kpc(redshift)

    if static_geometry is None:
        data = ellipse_source.data
        static_geometry = [
            (data["width"][n] / 2.0, data["height"][n] / 2.0, data["angle"][n])
            for n in range(len(data["x"]))
        ]

    readout_labels = []
    for n, label in enumerate(labels):
        semi_major, semi_minor, theta = static_geometry[n]
        readout = Label(
            x=10.0,
            y=0.0,  # overwritten by APERTURE_READOUT_POSITION_JS once laid out
            x_units="screen",
            y_units="screen",
            text=_aperture_readout_text(
                label, semi_major, semi_minor, theta, cd, arcsec_per_kpc
            ),
            text_font_size="11px",
            text_color=colors[n],
            text_baseline="top",       # hang below the anchor, so lines stack downward
            background_fill_color="white",
            background_fill_alpha=0.75,
            level="overlay",           # draw above the image and the glyphs
            name=f"aperture_readout_{n}",
        )
        fig.add_layout(readout)
        readout_labels.append(readout)

    position_callback = CustomJS(
        args={
            "fig": fig,
            "readout_labels": readout_labels,
            "top_margin": float(top_margin),
            "line_height": float(line_height),
        },
        code=APERTURE_READOUT_POS_JS,
    )
    fig.js_on_change("inner_height", position_callback)
    # Belt and braces: x_range "end" is already known to fire on first layout in
    # this plot (the loading-indicator callback relies on it), so the labels are
    # positioned even if inner_height lands before the callback is attached.
    fig.x_range.js_on_change("end", position_callback)

    if ellipse_source is not None:
        ellipse_source.js_on_change(
            "data",
            CustomJS(
                args={
                    "ellipse_source": ellipse_source,
                    "readout_labels": readout_labels,
                    "cd": cd,
                    "arcsec_per_kpc": arcsec_per_kpc,
                    "aperture_labels": list(labels),
                },
                code=APERTURE_LIVE_READOUT_JS,
            ),
        )

    return readout_labels


def plot_image_grid(image_dict, apertures=None):
    figures = []
    for survey, image in image_dict.items():
        fig = figure(
            title=survey,
            x_axis_label="",
            y_axis_label="",
            # plot_width=1000,
            # plot_height=1000,
            sizing_mode="scale_both",
        )
        fig = plot_image(fig, image)
        if apertures is not None:
            aperture = apertures.get(survey)
            if aperture is not None:
                wcs = WCS(image[0].header)
                aperture = apertures[survey].to_pixel(wcs)
                fig = plot_aperture(fig, aperture)
        figures.append(fig)

    plot = gridplot(figures, ncols=3, width=400, height=400)
    script, div = components(plot)
    return {"bokeh_cutout_script": script, "bokeh_cutout_div": div}


def plot_cutout_image(cutout=None, transient=None, global_aperture=None, local_aperture=None, editable=False):
    def generate_plot(fig, image_data):
        hide_loading_indicator = CustomJS(args=dict(), code="""
            document.getElementById('loading-indicator').style.display = "none";
        """)
        fig.x_range.js_on_change('end', hide_loading_indicator)
        plot_image(image_data, fig)
        script, div = components(fig)
        return {"bokeh_cutout_script": script, "bokeh_cutout_div": div}

    def generate_empty_plot(title):
        fig = figure(
            title=f"{title}",
            x_axis_label="",
            y_axis_label="",
            # plot_width=700,
            # plot_height=700,
            sizing_mode="scale_both",
        )
        fig.axis.visible = False
        fig.xgrid.visible = False
        fig.ygrid.visible = False
        image_data = np.zeros((500, 500))
        return generate_plot(fig, image_data)
    
    def add_apertures(fig, global_aperture, local_aperture, wcs, editable, transient):
        specs = []
        if global_aperture.exists():
            specs.append({
                "record": global_aperture[0],
                "aperture_type": "global",
                "line_color": "green",
                "legend_label": "Global Aperture",
                "readout_label": "Global",
            })
        if local_aperture.exists():
            specs.append({
                "record": local_aperture[0],
                "aperture_type": "local",
                "line_color": "blue",
                "legend_label": "Local Aperture",
                "readout_label": "Local",
            })

        if not specs:
            return

        readout_labels = [s["readout_label"] for s in specs]
        readout_colors = [s["line_color"] for s in specs]
        redshift = getattr(transient, "best_redshift", None)


        if editable:
            sources = plot_editable_apertures(
                fig,
                [
                    {
                        "aperture_id": s["record"].id,
                        "aperture_type": s["aperture_type"],
                        "sky_aperture": s["record"].sky_aperture,
                        "line_color": s["line_color"],
                        "legend_label": s["legend_label"],
                    }
                    for s in specs
                ],
                wcs,
            )
            add_aperture_readout(
                fig,
                wcs,
                redshift,
                labels=readout_labels,
                colors=readout_colors,
                ellipse_source=sources["ellipse_source"],
            )
        else:
            for s in specs:
                plot_aperture(
                    fig,
                    s["record"].sky_aperture,
                    wcs,
                    plotting_kwargs={
                        "fill_alpha": 0.1,
                        "line_color": s["line_color"],
                        "legend_label": s["legend_label"],
                    },
                )
            add_aperture_readout(
                fig,
                wcs,
                redshift,
                labels=readout_labels,
                colors=readout_colors,
                static_geometry=[
                    _pixel_geometry(s["record"].sky_aperture, wcs) for s in specs
                ],
            )

    # If there is no cutout data, generate an empty plot
    if cutout is None:
        return generate_empty_plot(title="No cutout selected")

    # Load image data from FITS file
    cutout_fits_path = cutout.fits.name
    local_tmp_path = os.path.join('/tmp', cutout_fits_path.strip('/').replace('/', '__'))
    object_key = os.path.join(settings.S3_BASE_PATH, cutout_fits_path.strip('/'))
    s3 = ObjectStore()
    if s3.object_exists(object_key):
        # Download FITS file local file cache
        s3.download_object(path=object_key, file_path=local_tmp_path)
    else:
        logger.error(f'''Data object "{object_key}" not found for missing data file "{cutout_fits_path}".''')
    # If the file is missing for some reason, generate an empty plot with an error title
    if not os.path.isfile(local_tmp_path):
        title = f'''Missing data file: "{os.path.basename(cutout_fits_path)}". '''
        title += '''Reprocess transient to regenerate missing data.'''
        return generate_empty_plot(title=title)
    try:
        with fits.open(local_tmp_path) as fits_file:
            image_data = fits_file[0].data
            wcs = WCS(fits_file[0].header)
    finally:
        # Delete FITS file from local file cache
        os.remove(local_tmp_path)

    title = cutout.filter
    fig = figure(
        title=f"{title}",
        x_axis_label="",
        y_axis_label="",
        # plot_width=500,
        # plot_height=int(np.shape(image_data)[0] / np.shape(image_data)[1] * 700),
        sizing_mode="scale_both",
    )
    fig.axis.visible = False
    fig.xgrid.visible = False
    fig.ygrid.visible = False

    transient_kwargs = {
        "legend_label": f"{transient.name}",
        "size": 30,
        "line_width": 2,
        "marker": "cross",
    }
    plot_position(
        transient, wcs, plotting_kwargs=transient_kwargs, plotting_func=fig.scatter
    )

    if transient.host is not None:
        host_kwargs = {
            "legend_label": f"Host: {transient.host.name}",
            "size": 25,
            "line_width": 2,
            "line_color": "red",
            "marker": "x",
        }
        plot_position(
            transient.host,
            wcs,
            plotting_kwargs=host_kwargs,
            plotting_func=fig.scatter,
        )
        
    add_apertures(
        fig,
        global_aperture,
        local_aperture,
        wcs,
        editable=editable,
        transient=transient,
    )
    return generate_plot(fig, image_data=image_data)


def plot_sed(transient=None, sed_results_file=None, type="", sed_modeldata_file=None, offset_sed_model=False):
    """
    Plot SED from aperture photometry.
    """

    try:
        obs = build_obs(transient, type, use_mag_offset=False)
    except ValueError:
        obs = {"filters": [], "maggies": [], "maggies_unc": []}
    except AssertionError:
        obs = {"filters": [], "maggies": [], "maggies_unc": []}

    # def maggies_to_asinh(x):
    #     """asinh magnitudes"""
    #     a = 2.50 * np.log10(np.e)
    #     mu = 35.0
    #     return -a * math.asinh((x / 2.0) * np.exp(mu / a)) + mu

    # def asinh_to_maggies(x):
    #     mu = 35.0
    #     a = 2.50 * np.log10(np.e)
    #     return np.array([2 * math.sinh((mu - x1) / a) * np.exp(-mu / a) for x1 in x])

    flux, flux_error, wavelength, filters, mag, mag_error = [], [], [], [], [], []
    for fl, f, fe in zip(obs["filters"], obs["maggies"], obs["maggies_unc"]):
        wavelength += [fl.wave_effective]
        flux += [maggies_to_mJy(f)]
        flux_error += [maggies_to_mJy(fe)]
        filters += [fl.name]
        mag += [-2.5 * np.log10(f)]
        mag_error += [1.086 * fe / f]

    fig = figure(
        title="",
        # max_width=600,
        sizing_mode="stretch_width",
        max_height=400,
        min_border=0,
        #    toolbar_location=None,
        x_axis_type="log",
        x_axis_label="Wavelength [angstrom]",
        y_axis_label="Flux [microjansky]",
    )

    if len(flux):
        fig.y_range = Range1d(-0.05 * np.max(flux), 1.5 * np.max(flux))
        fig.x_range = Range1d(np.min(wavelength) * 0.5, np.max(wavelength) * 1.5)
    else:
        fig.title = "Data Not Available"

    source = ColumnDataSource(
        data=dict(
            x=wavelength,
            y=flux,
            flux_error=flux_error,
            filters=filters,
            mag=mag,
            mag_error=mag_error,
        )
    )

    fig, p = plot_errorbar(
        fig,
        wavelength,
        flux,
        yerr=flux_error,
        point_kwargs={"size": 10, "legend_label": "data; MW E(B-V)-corrected"},
        error_kwargs={"width": 2},
        source=source,
    )

    # mouse-over for data
    TOOLTIPS = [
        ("flux (uJy)", "$y"),
        ("flux error (uJy)", "@flux_error"),
        ("wavelength", "$x"),
        ("band", "@filters"),
        ("mag (AB)", "@mag"),
        ("mag error (AB)", "@mag_error"),
    ]
    hover = HoverTool(renderers=[p], tooltips=TOOLTIPS, visible=False)
    fig.add_tools(hover)

    # second check on SED file
    # long-term shouldn't be necessary, just a result of debugging
    if sed_results_file is not None and os.path.exists(sed_results_file) and os.path.exists(sed_modeldata_file):
        logger.debug(f'''Loading results file "{sed_results_file}"...''')
        result, obs, _ = reader.results_from(sed_results_file, dangerous=False)
        logger.debug(f'''Loading model data file "{sed_modeldata_file}"...''')
        model_data = np.load(sed_modeldata_file, allow_pickle=True)

        # best = result["bestfit"]
        if transient.best_redshift < 0.015 and offset_sed_model:
            a = result["obs"]["redshift"] - 0.015 + 1
            mag_off = (
                cosmo.distmod(result["obs"]["redshift"]).value
                - cosmo.distmod(result["obs"]["redshift"] - 0.015).value
            )
            logger.debug(f"mag off: {mag_off}")
            fig.line(
                a * model_data["rest_wavelength"],
                maggies_to_mJy(model_data["spec"]) * 10 ** (0.4 * mag_off),
                legend_label="average model",
            )
            fig.line(
                a * model_data["rest_wavelength"],
                maggies_to_mJy(model_data["spec_16"]) * 10 ** (0.4 * mag_off),
                line_dash="dashed",
                legend_label="68% CI",
            )
            fig.line(
                a * model_data["rest_wavelength"],
                maggies_to_mJy(model_data["spec_84"]) * 10 ** (0.4 * mag_off),
                line_dash="dashed",
            )
        else:
            mag_off = 0
            a = result["obs"]["redshift"] + 1
            fig.line(
                a * model_data["rest_wavelength"],
                maggies_to_mJy(model_data["spec"]),
                legend_label="average model",
            )
            fig.line(
                a * model_data["rest_wavelength"],
                maggies_to_mJy(model_data["spec_16"]),
                line_dash="dashed",
                legend_label="68% CI",
            )
            fig.line(
                a * model_data["rest_wavelength"],
                maggies_to_mJy(model_data["spec_84"]),
                line_dash="dashed",
            )

        #  pre-SBI++ version: fig.line(a * best["restframe_wavelengths"], maggies_to_mJy(best["spectrum"]))
        if obs["filters"] is not None:
            try:
                pwave = [
                    Filter.objects.get(name=f).transmission_curve().wave_effective
                    for f in obs["filters"]
                ]
            except Exception:
                pwave = [f.wave_effective for f in obs["filters"]]

            if transient.best_redshift < 0.015 and offset_sed_model:
                fig.scatter(
                    pwave,
                    maggies_to_mJy(model_data["phot"]) * 10 ** (0.4 * mag_off),
                    size=10,
                )
            else:
                fig.scatter(pwave, maggies_to_mJy(model_data["phot"]), size=10)

    fig.legend.location = "top_left"

    script, div = components(fig)
    return {f"bokeh_sed_{type}_script": script, f"bokeh_sed_{type}_div": div, "fig": fig}


def plot_errorbar(
    figure,
    x,
    y,
    xerr=None,
    yerr=None,
    color="red",
    point_kwargs={},
    error_kwargs={},
    source=None,
):
    """
    Plot data points with error bars on a bokeh plot
    """

    # to do the mouse-over
    if source is not None:
        p = figure.scatter("x", "y", color=color, source=source, **point_kwargs)
    else:
        p = figure.scatter(x, y, color=color, source=source, **point_kwargs)

    if xerr:
        x_err_x = []
        x_err_y = []
        for px, py, err in zip(x, y, xerr):
            x_err_x.append((px - err, px + err))
            x_err_y.append((py, py))
        figure.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

    if yerr:
        y_err_x = []
        y_err_y = []
        for px, py, err in zip(x, y, yerr):
            y_err_x.append((px, px))
            y_err_y.append((py - err, py + err))
        figure.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)
    return figure, p


def plot_bar_chart(data_dict):
    x_label = ""
    y_label = "Transients"
    transient_numbers = list(data_dict.values())

    # bokeh 2.4.x bug where transients has to be string for LabelSet to work
    vals = pd.DataFrame(
        {
            "processing": list(data_dict.keys()),
            "transients": np.array(transient_numbers).astype(str).tolist(),
            "index": range(len(data_dict.values())),
        }
    )
    source = ColumnDataSource(vals)

    graph = figure(
        x_axis_label=x_label,
        y_axis_label=y_label,
        x_range=vals["processing"],
        y_range=Range1d(start=0, end=max(transient_numbers) * 1.1),
        sizing_mode="scale_both",
    )

    labels = LabelSet(
        x="index",
        y="transients",
        text="transients",
        source=source,
        level="glyph",
        y_offset=5,
        x_offset=64,
        text_align="center",
    )

    graph.vbar(source=source, x="processing", top="transients", bottom=0, width=0.7, color='#d15e00')
    graph.add_layout(labels)

    # displaying the model
    script, div = components(graph)
    return {"bokeh_cutout_script": script, "bokeh_cutout_div": div}


def plot_pie_chart(data_dict):
    data = (
        pd.Series(data_dict)
        .reset_index(name="value")
        .rename(columns={"index": "country"})
    )
    data["angle"] = data["value"] / data["value"].sum() * 2 * pi
    data["color"] = Category20[len(data)]

    p = figure(
        height=350,
        width=650,
        title="",
        toolbar_location=None,
        tools="hover",
        tooltips="@country: @value",
        x_range=(-0.5, 1.0),
    )

    p.wedge(
        x=0,
        y=1,
        radius=0.4,
        start_angle=cumsum("angle", include_zero=True),
        end_angle=cumsum("angle"),
        line_color="white",
        fill_color="color",
        legend_field="country",
        source=data,
    )

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None
    script, div = components(p)
    return {"bokeh_cutout_script": script, "bokeh_cutout_div": div}


def plot_timeseries():
    fig = figure(
        title="",
        width=700,
        height=400,
        min_border=0,
        toolbar_location=None,
        x_axis_type="log",
        x_axis_label="Time",
        y_axis_label="Number of Transients",
    )

    script, div = components(fig)
    return {
        "bokeh_processing_trends_script": script,
        "bokeh_processing_trends_div": div,
    }


def download_file_from_s3(file_path, object_key):
    s3 = ObjectStore()
    try:
        # Download SED results files to local file cache
        s3.download_object(path=object_key, file_path=file_path)
        assert os.path.isfile(file_path)
    except Exception as err:
        logger.error(f'''Error downloading SED file "{file_path}": {err}''')


def delete_cached_file(file_path):
    if not isinstance(file_path, str):
        return
    try:
        if os.path.isfile(file_path):
            os.remove(file_path)
    except Exception as err:
        logger.error(f'''Error deleting cached SED file "{file_path}": {err}''')


def temp_results_paths_from_canonical_path(canonical_path):
    sed_results_tmp_filepath = os.path.join('/tmp', canonical_path.strip('/').replace('/', '__'))
    sed_results_object_key = os.path.join(settings.S3_BASE_PATH, canonical_path.strip('/'))
    return sed_results_tmp_filepath, sed_results_object_key


def render_sed_plot(transient, scope):
    '''Generate a Bokeh plot of SED results'''

    assert scope in ["local", "global"]
    # Download the data files if they exist
    canonical_path = None
    sed_results_tmp_filepath = None
    sed_modeldata_tmp_filepath = None
    sed_obj = SEDFittingResult.objects.filter(transient=transient, aperture__type__exact=scope)
    offset_sed_model = False
    if sed_obj.exists():
        canonical_path = sed_obj[0].posterior.name
        sed_results_tmp_filepath, sed_results_object_key = temp_results_paths_from_canonical_path(canonical_path)
        sed_modeldata_tmp_filepath = sed_results_tmp_filepath.replace(".h5", "_modeldata.npz")
        sed_modeldata_object_key = sed_results_object_key.replace(".h5", "_modeldata.npz")
        download_file_from_s3(sed_results_tmp_filepath, sed_results_object_key)
        download_file_from_s3(sed_modeldata_tmp_filepath, sed_modeldata_object_key)
        if sed_obj[0].software_version is None or Version(sed_obj[0].software_version) <= Version('1.13.1'):
            offset_sed_model = True
    # Generate a SED plot using Bokeh
    plot = plot_sed(
        transient=transient,
        type=scope,
        sed_results_file=sed_results_tmp_filepath,
        sed_modeldata_file=sed_modeldata_tmp_filepath,
        offset_sed_model=offset_sed_model
    )
    # Purge temporary cached files
    delete_cached_file(sed_results_tmp_filepath)
    delete_cached_file(sed_modeldata_tmp_filepath)
    return {
        **plot,
        'canonical_path': canonical_path,
    }


def _read_host_spectrum_fits(local_fits_path):
    """
    Read a host spectrum FITS file saved by fetch_host_spectrum(): a
    SPARCL-derived binary table (extension "SPECTRUM" with wavelength/
    flux columns).
    """
    with fits.open(local_fits_path) as hdulist:
        flux_unit = hdulist[0].header.get('BUNIT', '1e-17 erg cm-2 s-1 AA-1')
        table = hdulist['SPECTRUM']
        wavelength = np.asarray(table.data['wavelength'], dtype=np.float64)
        flux = np.asarray(table.data['flux'], dtype=np.float64)
        wave_unit = table.columns['wavelength'].unit or 'AA'

    return {
        'wavelength': wavelength,
        'flux': flux,
        'wave_unit': wave_unit,
        'flux_unit': flux_unit,
    }


def plot_host_spectrum(host_spectrum=None, spectrum_file=None):
    """
    Plot the host galaxy spectrum downloaded by fetch_host_spectrum().
    """
    fig = figure(
        title="",
        sizing_mode="stretch_width",
        max_height=400,
        min_border=0,
        x_axis_label="Wavelength [Angstrom]",
        y_axis_label="Flux",
    )

    if spectrum_file is not None and os.path.exists(spectrum_file):
        spec = _read_host_spectrum_fits(spectrum_file)
        fig.yaxis.axis_label = f"Flux [{spec['flux_unit']}]"

        source = ColumnDataSource(data=dict(x=spec['wavelength'], y=spec['flux']))
        legend_label = f"{host_spectrum.source} spectrum" if host_spectrum else "spectrum"
        line = fig.line('x', 'y', source=source, legend_label=legend_label)

        TOOLTIPS = [
            ("wavelength", "$x"),
            ("flux", "$y"),
        ]
        hover = HoverTool(renderers=[line], tooltips=TOOLTIPS)
        fig.add_tools(hover)

        fig.legend.location = "top_left"
    else:
        fig.title = "Data Not Available"

    script, div = components(fig)
    return {"bokeh_host_spec_script": script, "bokeh_host_spec_div": div, "fig": fig}


def render_host_spectrum_plot(transient):
    '''Generate a Bokeh plot of the archival host galaxy spectrum'''

    canonical_path = None
    spectrum_tmp_filepath = None
    host_spectrum = None
    if transient.host is not None:
        host_spectrum_qs = HostSpectrum.objects.filter(host=transient.host)
        if host_spectrum_qs.exists():
            host_spectrum = host_spectrum_qs[0]
    if host_spectrum is not None and host_spectrum.spectrum_file:
        canonical_path = host_spectrum.spectrum_file.name
        spectrum_tmp_filepath, spectrum_object_key = temp_results_paths_from_canonical_path(canonical_path)
        download_file_from_s3(spectrum_tmp_filepath, spectrum_object_key)
    plot = plot_host_spectrum(
        host_spectrum=host_spectrum,
        spectrum_file=spectrum_tmp_filepath,
    )
    # Purge temporary cached file
    delete_cached_file(spectrum_tmp_filepath)
    return {
        **plot,
        'canonical_path': canonical_path,
        'host_spectrum': host_spectrum,
    }
