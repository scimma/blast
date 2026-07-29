import math
import os
from math import pi

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
from bokeh.models import CustomJS
from host.object_store import ObjectStore
from django.conf import settings
from bokeh.io import curdoc

# import extinction
# from bokeh.models import Circle
# from bokeh.models import Cross
# from bokeh.models import Ellipse
# from bokeh.models import Grid
# from bokeh.models import Legend
# from bokeh.models import LinearAxis
# from bokeh.models import LogColorMapper
# from bokeh.models import Plot
# from bokeh.models import Scatter
# from bokeh.plotting import show
# from host.catalog_photometry import filter_information
# from host.host_utils import survey_list
# from host.photometric_calibration import mJy_to_maggies
# from host.prospector import build_model

# from .models import Aperture

from host.log import get_logger
logger = get_logger(__name__)

########################################## MY ADDED CUSTOMJS ##########################################
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
        const majorIdx = 2 * n;
        const minorIdx = 2 * n + 1;

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

        const minorUnitX = -Math.sin(theta);
        const minorUnitY = Math.cos(theta);

        const minorDx = handles.x[minorIdx] - cx;
        const minorDy = handles.y[minorIdx] - cy;

        // Project the dragged minor handle onto the true minor axis so it
        // can't drift away from a perpendicular position.
        const projectedMinorRadius = Math.abs(
            minorDx * minorUnitX +
            minorDy * minorUnitY
        );

        // Keep the aperture mathematically consistent: semi-major >= semi-minor.
        const semiMinor = Math.min(
            semiMajor,
            Math.max(minimum_radius, projectedMinorRadius),
        );

        ellipses.width[n] = 2.0 * semiMajor;
        ellipses.height[n] = 2.0 * semiMinor;
        ellipses.angle[n] = theta;

        // Snap both controls back onto their exact axes.
        handles.x[majorIdx] = cx + semiMajor * Math.cos(theta);
        handles.y[majorIdx] = cy + semiMajor * Math.sin(theta);

        handles.x[minorIdx] = cx + semiMinor * minorUnitX;
        handles.y[minorIdx] = cy + semiMinor * minorUnitY;

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


########################################## ORIGINAL CODE ###############################################################

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
    theta_rad = aperture.theta
    x, y = aperture.positions
    plot_dict = {
        "x": x,
        "y": y,
        "width": aperture.a * 2,
        "height": aperture.b * 2,
        "angle": theta_rad.value,
        "fill_color": "#cab2d6",
        "fill_alpha": 0.1,
        "line_width": 4,
    }

    plot_dict = {**plot_dict, **plotting_kwargs}
    figure.ellipse(**plot_dict)
    return figure

def plot_editable_apertures(figure, apertures, wcs, minimum_radius=1.0):
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

    handle_x, handle_y, handle_color, handle_axis = [], [], [], []
    aperture_ids, aperture_types = [], []

    for ap in apertures:
        pixel_aperture = ap["sky_aperture"].to_pixel(wcs)
        position = np.asarray(pixel_aperture.positions, dtype=float)

        center_x = float(position[0])
        center_y = float(position[1])
        semi_major = float(pixel_aperture.a)
        semi_minor = float(pixel_aperture.b)
        theta = float(pixel_aperture.theta.value)

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

        major_handle_x = center_x + semi_major * math.cos(theta)
        major_handle_y = center_y + semi_major * math.sin(theta)
        minor_handle_x = center_x - semi_minor * math.sin(theta)
        minor_handle_y = center_y + semi_minor * math.cos(theta)

        # Handles inherit the color of their own aperture so it's visually
        # obvious which pair of handles controls which ellipse.
        handle_x += [major_handle_x, minor_handle_x]
        handle_y += [major_handle_y, minor_handle_y]
        handle_color += [ap["line_color"], ap["line_color"]]
        handle_axis += ["major", "minor"]

        aperture_ids.append(ap["aperture_id"])
        aperture_types.append(ap["aperture_type"])

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

    # Used only to prevent an infinite callback loop when the JavaScript
    # callback snaps the handles back onto their exact axes.
    callback_state = ColumnDataSource(data={"updating": [False]})

    ellipse_renderer = figure.ellipse(
        x="x",
        y="y",
        width="width",
        height="height",
        angle="angle",
        source=ellipse_source,
        fill_color="#cab2d6",
        fill_alpha=0.1,
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
    )

    # A single shared tool drives every editable aperture on this figure.
    handle_tool = PointDrawTool(renderers=[handle_renderer], add=False)
    figure.add_tools(handle_tool)

    handle_callback = CustomJS(
        args={
            "ellipse_source": ellipse_source,
            "handle_source": handle_source,
            "callback_state": callback_state,
            "aperture_ids": aperture_ids,
            "aperture_types": aperture_types,
            "minimum_radius": float(minimum_radius),
        },
        code=EDITABLE_APERTURES_HANDLE_JS,
    )

    # PointDrawTool modifies handle_source continuously while the pointer
    # moves, so this callback runs repeatedly throughout each drag.
    handle_source.js_on_change("data", handle_callback)

    return {
        "ellipse_source": ellipse_source,
        "handle_source": handle_source,
        "ellipse_renderer": ellipse_renderer,
        "handle_renderer": handle_renderer,
        "handle_tool": handle_tool,
    }


########################################## ORIGINAL CODE ###############################################################
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
    
    def add_apertures(fig, global_aperture, local_aperture, wcs, editable):
        specs = []
        if global_aperture.exists():
            specs.append({
                "record": global_aperture[0],
                "aperture_type": "global",
                "line_color": "green",
                "legend_label": "Global Aperture",
            })
        if local_aperture.exists():
            specs.append({
                "record": local_aperture[0],
                "aperture_type": "local",
                "line_color": "blue",
                "legend_label": "Local Aperture",
            })

        if not specs:
            return

        if editable:
            plot_editable_apertures(
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
    )
    return generate_plot(fig, image_data=image_data)


def plot_sed(transient=None, sed_results_file=None, type="", sed_modeldata_file=None):
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
        x_axis_label="Wavelength [Angstrom]",
        y_axis_label="Flux",
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
        if transient.best_redshift < 0.015:
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

            if transient.best_redshift < 0.015:
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
    if sed_obj.exists():
        canonical_path = sed_obj[0].posterior.name
        sed_results_tmp_filepath, sed_results_object_key = temp_results_paths_from_canonical_path(canonical_path)
        sed_modeldata_tmp_filepath = sed_results_tmp_filepath.replace(".h5", "_modeldata.npz")
        sed_modeldata_object_key = sed_results_object_key.replace(".h5", "_modeldata.npz")
        download_file_from_s3(sed_results_tmp_filepath, sed_results_object_key)
        download_file_from_s3(sed_modeldata_tmp_filepath, sed_modeldata_object_key)
    # Generate a SED plot using Bokeh
    plot = plot_sed(
        transient=transient,
        type=scope,
        sed_results_file=sed_results_tmp_filepath,
        sed_modeldata_file=sed_modeldata_tmp_filepath,
    )
    # Purge temporary cached files
    delete_cached_file(sed_results_tmp_filepath)
    delete_cached_file(sed_modeldata_tmp_filepath)
    return {
        **plot,
        'canonical_path': canonical_path,
    }
