import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DEPTHFLOW_DATA", os.path.join(ROOT, "data"))
RESULTS = os.path.join(ROOT, "results")


def data(name):
    return os.path.join(DATA, name)


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(obj, open(os.path.join(RESULTS, name + ".json"), "w"), indent=1)
    print("saved", name, flush=True)


def load(name):
    return json.load(open(os.path.join(RESULTS, name + ".json")))
