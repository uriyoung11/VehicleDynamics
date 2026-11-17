"""D7(Stanford 2025 VW Golf) CSV를 data/d7/에 받는다. 코랩·새 컴퓨터용.

  python colab/get_d7.py

파일 목록은 저장소 메타데이터(purl.stanford.edu/hh613qz0317.xml)에서 읽고,
https://stacks.stanford.edu/file/druid:hh613qz0317/<경로> 에서 받는다. CC BY-SA 4.0.
"""
import os, re, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "d7")
UA = {"User-Agent": "VehicleDynamicsResearch/0.1 (academic project)"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = get("https://purl.stanford.edu/hh613qz0317.xml").decode("utf-8", "replace")
    files = re.findall(r'<file id="(VehicleDynamicsDataset/[^"]+\.csv)"', meta)
    print(len(files), "files")

    def one(p):
        dst = os.path.join(OUT, os.path.basename(p))
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            open(dst, "wb").write(get("https://stacks.stanford.edu/file/druid:hh613qz0317/" + p))
        return os.path.basename(p)

    with ThreadPoolExecutor(6) as ex:
        for name in ex.map(one, files):
            print("ok", name)


if __name__ == "__main__":
    main()
