import csv
import json
import sys
import time
from urllib.request import Request
from urllib.request import urlopen
from urllib.error import HTTPError


def post_transient_from_csv(path_to_input_csv: str, base_url: str) -> None:
    """
    Post transients from csv file to blast for processing.

    Parameters
        path_to_input_csv (str): path to input transient csv file.
        base_url (str): base url to the api
    returns
        None, prints the status of each posted transient
    """
    with open(path_to_input_csv, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for transient in reader:
            ra, dec = transient["ra"], transient["dec"]
            name = transient["name"]
            post_url = f"{base_url}name={name}&ra={ra}&dec={dec}"
            try:
                response = urlopen(Request(post_url, method="POST"))
                data = json.loads(response.read())
                post_message = data.get("message", "no message returned by blast")
                print(f"{post_message}")
            except HTTPError as e:
                print(e.code)
                print(e.read())
            except Exception as e:
                print(f"{name}: {e}")


def download_data_snapshot(
    path_to_input_csv: str, path_to_output_csv: str, base_url: str
) -> None:
    """
    Downloads snapshot of data for transients

    Parameters
        path_to_input_csv (str): path to input transient csv file.
        base_url (str): base url to the api
    Returns
        None
    """
    payloads = []
    with open(path_to_input_csv, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for transient in reader:
            transient_name = transient["name"]
            post_url = f"{base_url}{transient_name}?format=json"
            response = urlopen(Request(post_url, method="GET"))
            data = json.loads(response.read())
            payloads.append(data)

    with open(path_to_output_csv, "w", newline="") as csv_file:
        fieldnames = payloads[0].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for payload in payloads:
            writer.writerow(payload)


def transient_processing_progress(path_to_output_csv: str) -> float:
    """
    Calculates the processing status of a batch of transients

    Parameters
        path_to_output_csv (str): Path to output csv file.
    Returns:
        processed_progress (float): Fraction of transients that have been
        completed (processed or blocked)
    """

    processing, completed = 0, 0
    with open(path_to_output_csv, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for transient in reader:
            progress = int(transient["transient_progress"])
            # TODO: This progress monitor is not ideal, because if a transient fails to begin (0) or the workflow
            #       fails to complete (<100), the script will idle indefinitely.
            if progress >= 0:
                processing = +1
            elif progress == 100:
                completed = +1

    return completed / (processing + completed)


if __name__ == "__main__":
    import os
    localhost = f'''http://{os.getenv('WEB_APP_HOST')}:{os.getenv('WEB_APP_PORT')}'''
    post_endpoint = "/api/transient/post/"
    get_endpoint = "/api/transient/get/"

    input_csv = str(sys.argv[1])
    post_transient_from_csv(input_csv, f"{localhost}{post_endpoint}")
    download_data_snapshot(input_csv, "/results.csv", f"{localhost}{get_endpoint}")
    batch_progress = transient_processing_progress("/results.csv")

    while batch_progress < 1.0:
        print(batch_progress)
        time.sleep(10)
        download_data_snapshot(input_csv, "/results.csv", f"{localhost}{get_endpoint}")
        batch_progress = transient_processing_progress("/results.csv")
