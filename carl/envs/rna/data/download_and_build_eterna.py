from tqdm import tqdm
import requests


def _download_dataset_from_http(url, download_path):
    response = requests.get(url, stream=True)
    with open(download_path, "wb+") as dataset_file:
        progress_bar = tqdm(
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            total=int(response.headers["Content-Length"]),
        )
        for data in tqdm(response.iter_content()):
            progress_bar.update(len(data))
            dataset_file.write(data)


def download_eterna(download_path):
    eterna_url = "https://ars.els-cdn.com/content/image/1-s2.0-S0022283615006567-mmc5.txt"
    _download_dataset_from_http(eterna_url, download_path)


def extract_secondarys(download_path, dump_path):
    with open(download_path) as input:
        parsed = list(zip(*(line.strip().split("\t") for line in input)))

    secondarys = parsed[4][1:]

    with open(dump_path, "w") as data_file:
        for structure in secondarys:
            if structure[-1] == "_":  # Weird dataset bug
                structure = structure[:-1]
            data_file.write(f"{structure}\n")


if __name__ == "__main__":
    download_path = f"data/eterna/raw/eterna_raw.txt"
    dump_path = f"data/eterna/interim/eterna.txt"
    download_eterna(download_path)
    extract_secondarys(download_path, dump_path)
