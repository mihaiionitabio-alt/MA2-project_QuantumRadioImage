"""Download and verify the public EHT M87 data and official imaging pipeline."""

from __future__ import annotations

import hashlib
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "eht_m87"
PIPELINE_DIR = ROOT / "external" / "eht-m87-imaging"
PIPELINE_REPOSITORY = "https://github.com/eventhorizontelescope/2019-D01-02.git"
CYVERSE_BASE = (
    "https://de.cyverse.org/anon-files//iplant/home/shared/commons_repo/"
    "curated/EHTC_FirstM87Results_Apr2019/uvfits"
)
FILES = {
    "SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits": (
        "697af2bb3bbf732115108ffefacd3e59e307f38fe685c3c4579146b0bd661298"
    ),
    "SR1_M87_2017_101_hi_hops_netcal_StokesI.uvfits": (
        "618c3019f60e88268980267a9db68f638b379a85842d83c06003d28550d191f5"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(filename: str, expected_hash: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    destination = DATA_DIR / filename
    if destination.exists() and sha256(destination) == expected_hash:
        print(f"Verified existing file: {destination}")
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{CYVERSE_BASE}/{filename}"
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, temporary)
    actual_hash = sha256(temporary)
    if actual_hash != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {filename}: {actual_hash} != {expected_hash}"
        )
    temporary.replace(destination)
    print(f"Downloaded and verified: {destination}")
    return destination


def clone_pipeline() -> Path:
    pipeline_script = (
        PIPELINE_DIR / "eht-imaging" / "eht-imaging_pipeline.py"
    )
    if pipeline_script.exists():
        print(f"Found official pipeline: {pipeline_script}")
        return pipeline_script

    PIPELINE_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            PIPELINE_REPOSITORY,
            str(PIPELINE_DIR),
        ],
        check=True,
    )
    if not pipeline_script.exists():
        raise FileNotFoundError(
            "The clone completed but the eht-imaging pipeline was not found."
        )
    print(f"Downloaded official pipeline: {pipeline_script}")
    return pipeline_script


def main() -> int:
    for filename, expected_hash in FILES.items():
        download_file(filename, expected_hash)
    clone_pipeline()
    print("EHT research inputs are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
