import json

from utils import truth_from_csv

if __name__ == '__main__':
    in_csv = "/blender/storage/datasets/vg_smoke/valid/basic_inst/basic-instinct_truth.csv"
    out_json = "/blender/storage/datasets/vg_smoke/valid/basic_inst/basic-instinct_truth.csv.jsonl"

    by_frame = truth_from_csv(truth_csv=in_csv, num_classes=2)

    with open(out_json, "w") as _f:
        for fn, r in enumerate(by_frame):
            v = (fn, r.tolist())
            s = json.dumps(v)
            print(s)
            _f.write("%s\n" % s)
