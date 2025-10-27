"""
For licensing see accompanying LICENSE file.
Writen by: ian
"""
import os
import requests

from utils.logger import logger


COCO128 = ""


def _download_file(url_path: str, dest_loc: str) -> None:
    """
    Helper function to download a file with proxy (used when file fails)
    """
    response = requests.get(url_path, stream=True)
    if response.status_code == 403:
        # try with the HTTP/HTTPS proxy from ENV
        proxies = {
            "https": os.environ.get("HTTPS_PROXY", None),
            "http": os.environ.get("HTTP_PROXY", None),
        }
        response = requests.get(url_path, stream=True, proxies=proxies)

    if response.status_code == 200:
        with open(dest_loc, "wb") as f:
            f.write(response.raw.read())
    else:
        logger.error("Unable to download file {}".format(url_path))