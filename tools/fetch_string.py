"""Download the STRING v12.0 E. coli K-12 network and write the filtered
copies kept in data/ (links with a combined score of at least 700, each
undirected link once, and the protein names).

    python tools/fetch_string.py
"""
import gzip
import hashlib
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
BASE = "https://stringdb-downloads.org/download"
FILES = {
    "links": (f"{BASE}/protein.links.v12.0/511145.protein.links.v12.0.txt.gz",
              "e60afb17840044fe25e10572291e07d8715899238e10048dd6e95e9ed109e19c"),
    "info": (f"{BASE}/protein.info.v12.0/511145.protein.info.v12.0.txt.gz",
             "f86c3ec473f9e27f08cb8341063e8173206cd0d0970b5764623a7fe73c3b1076"),
}


def fetch(url, sha256):
    raw = urllib.request.urlopen(url).read()
    if hashlib.sha256(raw).hexdigest() != sha256:
        print(f"Warning: {url} differs from the copy used for data/")
    return gzip.decompress(raw).decode().splitlines()


lines = fetch(*FILES["links"])
with open(DATA / "ecoli_string_ppi.csv", "w") as out:
    out.write("protein1,protein2,combined_score\n")
    for line in lines[1:]:
        a, b, s = line.split()
        if int(s) >= 700 and a < b:
            out.write(f"{a},{b},{s}\n")

lines = fetch(*FILES["info"])
with open(DATA / "ecoli_string_proteins.csv", "w") as out:
    out.write("string_id,name,size\n")
    for line in lines[1:]:
        p = line.split("\t")
        out.write(f"{p[0]},{p[1]},{p[2]}\n")
