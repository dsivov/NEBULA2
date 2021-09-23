# source: https://gist.github.com/getnamo/239aef074f6fe5c898fef66b53b0f40a
import argparse

import requests
from tqdm import tqdm
import os


def download_file_from_google_drive(file_id, destination, show_pbar=True):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : file_id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : file_id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination, show_pbar=show_pbar)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination, show_pbar=True):
    CHUNK_SIZE = 32768
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    pbar = tqdm(total=len(response.content), disable=not show_pbar)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
            pbar.update(len(chunk))
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Downloads a file from google drive.')
    parser.add_argument('gdrive_id', metavar='id',
                        help='The Goodle Drive ID of the file. Can be found in the file URL.')
    parser.add_argument('--output', '-o',
                        help='The output path for the downloaded file.')
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(os.getcwd(), args.gdrive_id)

    download_file_from_google_drive(args.gdrive_id, args.output)
