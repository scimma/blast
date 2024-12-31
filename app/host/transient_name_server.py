""""
Functions to download and ingest transients from the Transient Name Server (TNS)
"""
import io
import json
import os
import time
import zipfile
from collections import OrderedDict
from io import StringIO

import pandas as pd
import requests

from .models import Transient


def get_tns_credentials():
    """
    Retrieves TNS credentials from environment variables
    """
    credentials = ["TNS_BOT_API_KEY", "TNS_BOT_NAME", "TNS_BOT_ID"]

    for credential in credentials:
        credential_value = os.environ.get(credential)
        if credential_value is None:
            raise ValueError(f"{credential} not defined in environment")

    return {credential: os.environ[credential] for credential in credentials}


# @log_resource_call(resource_name='Transient name server')
def query_tns(data, headers, search_url):
    """
    Query the TNS server
    """

    response = requests.post(search_url, headers=headers, data=data)
    response_json = json.loads(response.text)

    response_message = response_json.get("id_message")
    response_id_code = response_json.get("id_code")

    response_status_good = response_id_code == 200
    data = response_json.get("data", {}) if response_status_good else []
    response_reset_time = response.headers['x-rate-limit-reset']

    response_return = {
        "response_message": response_message,
        "response_id_code": response_id_code,
        "data": data,
        "response_reset_time": response_reset_time,
    }
    return response_return


def rate_limit_query_tns(data, headers, search_url):
    """
    Query TNS but wait if we have reached too many api requests.
    """
    response = query_tns(data, headers, search_url)
    too_many_requests = response["response_id_code"] == 429
    while too_many_requests:
        time_util_rest = response["response_reset_time"]
        time.sleep(time_util_rest + 1)
        response = query_tns(data, headers, search_url)
        too_many_requests = response["response_id_code"] == 429
    return response["data"]


def get_transients_from_tns(time_after, sandbox=False, tns_credentials=None):
    """
    Gets transient data from TNS for all transients with public
    timestamp after time_after.

    Args:
        time_after (datetime.datetime):  Time to search the transient name
            server for new transients.
        sandbox (bool): If true uses the transient name server sandbox API,
            else uses the live transisent name server API
        tns_credentials (dict): Transient name server credentials, need to have
            the keys, TNS_BOT_ID, TNS_BOT_NAME, and TNS_BOT_API_KEY.
    Returns:
        (list): List of Transients retrieved transients from the transient name
            server.
    """

    tns_bot_id = tns_credentials["TNS_BOT_ID"]
    tns_bot_name = tns_credentials["TNS_BOT_NAME"]
    tns_bot_api_key = tns_credentials["TNS_BOT_API_KEY"]

    headers = build_tns_header(tns_bot_id, tns_bot_name)

    entry = "sandbox" if sandbox else "www"
    tns_api_url = f"https://{entry}.wis-tns.org/api/get"

    search_tns_url = build_tns_url(tns_api_url, mode="search")
    get_tns_url = build_tns_url(tns_api_url, mode="get")

    search_data = build_tns_search_query_data(tns_bot_api_key, time_after)
    transients = rate_limit_query_tns(search_data, headers, search_tns_url)

    blast_transients = []

    for transient in transients:
        get_data = build_tns_get_query_data(tns_bot_api_key, transient)
        tns_transient = rate_limit_query_tns(get_data, headers, get_tns_url)
        blast_transient = tns_to_blast_transient(tns_transient)
        blast_transients.append(blast_transient)

    return blast_transients


def get_transients_from_tns_by_name(
    transient_list, sandbox=False, tns_credentials=None
):
    """
    Gets transient data from TNS for all transients with public
    timestamp after time_after.

    Args:
        time_after (datetime.datetime):  Time to search the transient name
            server for new transients.
        sandbox (bool): If true uses the transient name server sandbox API,
            else uses the live transisent name server API
        tns_credentials (dict): Transient name server credentials, need to have
            the keys, TNS_BOT_ID, TNS_BOT_NAME, and TNS_BOT_API_KEY.
    Returns:
        (list): List of Transients retrieved transients from the transient name
            server.
    """

    if tns_credentials is None:
        tns_credentials = get_tns_credentials()

    tns_bot_id = tns_credentials["TNS_BOT_ID"]
    tns_bot_name = tns_credentials["TNS_BOT_NAME"]
    tns_bot_api_key = tns_credentials["TNS_BOT_API_KEY"]

    headers = build_tns_header(tns_bot_id, tns_bot_name)

    entry = "sandbox" if sandbox else "www"
    tns_api_url = f"https://{entry}.wis-tns.org/api/get"

    get_tns_url = build_tns_url(tns_api_url, mode="get")

    transients = []
    for t in transient_list:
        transients += [{"objname": t, "objid": t}]

    blast_transients = []

    for transient in transients:
        get_data = build_tns_get_query_data(tns_bot_api_key, transient)
        tns_transient = rate_limit_query_tns(get_data, headers, get_tns_url)
        blast_transient = tns_to_blast_transient(tns_transient)
        blast_transients.append(blast_transient)

    return blast_transients


def tns_to_blast_transient(tns_transient):
    """Convert transient name server transient into blast transient data model.

    Args:
        tns_transient (Dict): Dictionary containing transient name server
            transient information.
    Returns:
        blast_transient (Transient): Transient object with the
            tns_transient data.
    """
    blast_transient = Transient(
        name=tns_transient["objname"],
        tns_id=tns_transient["objid"],
        ra_deg=tns_transient["radeg"],
        dec_deg=tns_transient["decdeg"],
        tns_prefix=tns_transient["name_prefix"],
        public_timestamp=tns_transient["discoverydate"],
        spectroscopic_class=tns_transient["object_type"]["name"],
        redshift=tns_transient["redshift"],
    )
    return blast_transient


def tns_staging_blast_transient(tns_transient):
    """Convert transient name server staging transient into blast transient
    data model.

    Args:
        tns_transient (Dict): Dictionary containing transient name server
            transient information.
    Returns:
        blast_transient (Transient): Transient object with the
            tns_transient data.
    """
    blast_transient = Transient(
        name=tns_transient["name"],
        tns_id=tns_transient["objid"],
        ra_deg=tns_transient["ra"],
        dec_deg=tns_transient["declination"],
        tns_prefix=tns_transient["name_prefix"],
        public_timestamp=tns_transient["discoverydate"],
        spectroscopic_class=tns_transient["type"],
    )
    return blast_transient


def update_blast_transient(blast_transient, tns_staging_transient):
    """
    Updates blast transient with information from the tns_csv staging
    """
    mapping = {
        "ra_deg": "ra",
        "dec_deg": "declination",
        "tns_prefix": "name_prefix",
        "public_timestamp": "discoverydate",
        "spectroscopic_class": "type",
    }

    names_match = blast_transient.name == tns_staging_transient["name"]
    ids_match = blast_transient.tns_id == tns_staging_transient["objid"]

    if names_match and ids_match:
        for blast_attribute, staging_attribute in mapping.items():
            blast_value = getattr(blast_transient, blast_attribute)
            staging_value = tns_staging_transient[staging_attribute]
            values_match = blast_value == staging_value
            if values_match:
                pass
            else:
                setattr(blast_transient, blast_attribute, staging_value)

        blast_transient.save()


def build_tns_header(tns_bot_id, tns_bot_name):
    """
    Builds the TNS header dictionary.

    Args:
        tns_bot_id (int): Transient name server bot id.
        tns_bot_name (str): Transient name server bot name.
    Returns:
        (dict): Transient name server header dictionary.
    """
    tns_marker = (
        f'tns_marker{{"tns_id": "{int(tns_bot_id)}",'
        f'"type": "bot", "name": "{tns_bot_name}"}}'
    )
    return {"User-Agent": tns_marker}


def build_tns_query_data(tns_bot_api_key, data_obj):
    """
    Builds tns search data dictionary.

    Args:
        tns_bot_api_key (str): Transient name server bot api key.
        data_obj (list): List of data representing the tns query.
    Returns:
        (dict): Transient name server query data.

    """
    data_obj = OrderedDict(data_obj)
    return {"api_key": tns_bot_api_key, "data": json.dumps(data_obj)}


def build_tns_url(tns_api_url, mode=None):
    """
    Builds the url to the tns api service

    Args:
        tns_api_url (str): URL of the Transient name server API.
        mode (str): Which endpoint to access the API. Options are search and get
    Returns:
        (str) Full transient name server api url
    """
    if mode == "search":
        url_end_point = "/search"
    elif mode == "get":
        url_end_point = "/object"
    else:
        raise ValueError("Mode invalid, provide a valid mode (search or get)")
    return tns_api_url + url_end_point


def build_tns_get_query_data(tns_bot_api_key, transient):
    """
    Build the the get query data for a TNS transient.

    Args:
        tns_bot_api_key (str): Transient name server bot api key.
        transient (dict): Transient name server transient information.
    Returns:
        (dict): Transient name server query data.

    """
    get_obj = [
        ("objname", transient["objname"]),
        ("objid", transient["objid"]),
        ("photometry", "0"),
        ("spectra", "0"),
    ]
    return build_tns_query_data(tns_bot_api_key, get_obj)


def build_tns_search_query_data(tns_bot_api_key, time_after):
    """
    Build the the search query to find tns transients with a public timestamp
    after time after.

    Args:
        tns_bot_api_key (str): Transient name server bot api key.
        time_after (datetime.datetime): Time to search the transient name
            server for new transients.
    Returns:
        (dict): Transient name server query data.

    """
    search_obj = [("public_timestamp", time_after.isoformat())]
    return build_tns_query_data(tns_bot_api_key, search_obj)


def get_daily_tns_staging_csv(date, tns_credentials=None, save_dir=""):
    """
    Gets the daily staging csv from TNS that contains all transients added and
    modified at the specified date.
    """
    file_name = f"tns_public_objects_{date}.csv"

    if os.path.exists(f"{save_dir}/{file_name}"):
        staging_data = pd.read_csv(f"{save_dir}/{file_name}", header=1)
    else:
        tns_staging_url = "https://www.wis-tns.org/system/files/tns_public_objects/"
        tns_staging_url += f"{file_name}.zip"

        csv_path = get_tns_zipped_csv(
            url=tns_staging_url,
            file_name=file_name,
            tns_credentials=tns_credentials,
            save_dir=save_dir,
        )
        staging_data = pd.read_csv(csv_path, header=1)
    return staging_data


def get_tns_zipped_csv(url=None, file_name=None, tns_credentials=None, save_dir=""):
    """
    Downloads and saves zipped csv file from TNS.

    Returns the file name
    """

    tns_bot_id = tns_credentials["TNS_BOT_ID"]
    tns_bot_name = tns_credentials["TNS_BOT_NAME"]
    tns_bot_api_key = tns_credentials["TNS_BOT_API_KEY"]

    headers = build_tns_header(tns_bot_id, tns_bot_name)
    data = {"api_key": tns_bot_api_key}
    response = requests.post(url, headers=headers, data=data, stream=True)
    z = zipfile.ZipFile(io.BytesIO(response.content))
    file_path = f"{save_dir}/{file_name}"
    z.extractall(save_dir)
    return file_path


def tns_staging_file_date_name(date):
    """
    Gets the TNS file name for a staging csv for a given date.
    """
    return f"{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}"
