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
# from bokeh.models import ColumnDataSource
from bokeh.models import HoverTool
from bokeh.models import LabelSet
from bokeh.models import Range1d
from bokeh.palettes import Category20
from bokeh.plotting import ColumnDataSource
from bokeh.plotting import figure
from bokeh.transform import cumsum
from host.models import Filter
from host.photometric_calibration import maggies_to_mJy
from host.prospector import build_obs
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


def plot_cutout_image(cutout=None, transient=None, global_aperture=None, local_aperture=None):
    def generate_plot(fig, image_data):
        hide_loading_indicator = CustomJS(args=dict(), code="""
            document.getElementById('loading-indicator').style.display = "none";
        """)
        fig.x_range.js_on_change('end', hide_loading_indicator)
        plot_image(image_data, fig)

        script, div = components(fig)
        return {"bokeh_cutout_script": script, "bokeh_cutout_div": div}

    # If there is no cutout data, generate an empty plot
    if cutout is None:
        title = "No cutout selected"
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

    # Load image data from FITS file
    local_fits_path = cutout.fits.name
    if not os.path.isfile(local_fits_path):
        s3 = ObjectStore()
        object_key = os.path.join(settings.S3_BASE_PATH, local_fits_path.strip('/'))
        if s3.object_exists(object_key):
            # Download FITS file local file cache
            s3.download_object(path=object_key, file_path=local_fits_path)
        else:
            logger.error(f'''Data object "{object_key}" not found for missing data file "{local_fits_path}".''')
    assert os.path.isfile(local_fits_path)
    with fits.open(local_fits_path) as fits_file:
        image_data = fits_file[0].data
        wcs = WCS(fits_file[0].header)
    # Delete FITS file from local file cache
    os.remove(local_fits_path)

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

    if global_aperture.exists():
        plot_aperture(
            fig,
            global_aperture[0].sky_aperture,
            wcs,
            plotting_kwargs={
                "fill_alpha": 0.1,
                "line_color": "green",
                "legend_label": f"Global Aperture ({title})",
            },
        )

    if local_aperture.exists():
        plot_aperture(
            fig,
            local_aperture[0].sky_aperture,
            wcs,
            plotting_kwargs={
                "fill_alpha": 0.1,
                "line_color": "blue",
                "legend_label": "Local Aperture",
            },
        )
    return generate_plot(fig, image_data=image_data)


def plot_sed(transient=None, sed_results_file=None, type=""):
    """
    Plot SED from aperture photometry.
    """

    try:
        obs = build_obs(transient, type, use_mag_offset=False)
    except ValueError:
        obs = {"filters": [], "maggies": [], "maggies_unc": []}
    except AssertionError:
        obs = {"filters": [], "maggies": [], "maggies_unc": []}

    def maggies_to_asinh(x):
        """asinh magnitudes"""
        a = 2.50 * np.log10(np.e)
        mu = 35.0
        return -a * math.asinh((x / 2.0) * np.exp(mu / a)) + mu

    def asinh_to_maggies(x):
        mu = 35.0
        a = 2.50 * np.log10(np.e)
        return np.array([2 * math.sinh((mu - x1) / a) * np.exp(-mu / a) for x1 in x])

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

    # remove the loading spinner
    hide_loading_indicator = CustomJS(args=dict(), code=f"""
            document.getElementById('loading-indicator-sed-{type}').style.display = "none";
            document.getElementById('loading-indicator-sed-inf-{type}').style.display = "none";
        """)
    curdoc().js_on_event("document_ready", hide_loading_indicator)

    # second check on SED file
    # long-term shouldn't be necessary, just a result of debugging
    if sed_results_file is not None and os.path.exists(
        sed_results_file.replace(".h5", "_modeldata.npz")
    ):
        result, obs, _ = reader.results_from(sed_results_file, dangerous=False)
        model_data = np.load(
            sed_results_file.replace(".h5", "_modeldata.npz"), allow_pickle=True
        )

        # best = result["bestfit"]
        if transient.best_redshift < 0.015:
            a = result["obs"]["redshift"] - 0.015 + 1
            mag_off = (
                cosmo.distmod(result["obs"]["redshift"]).value
                - cosmo.distmod(result["obs"]["redshift"] - 0.015).value
            )
            print(f"mag off: {mag_off}")
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
    return {f"bokeh_sed_{type}_script": script, f"bokeh_sed_{type}_div": div}


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
