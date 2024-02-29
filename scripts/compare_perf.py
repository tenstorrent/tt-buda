# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from elasticsearch import Elasticsearch
import tabulate

# add project root to search path
project_root_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(project_root_path)

from third_party.confidential_keys.elastic_search import ES_ENDPOINT, ES_USERNAME, ES_PASSWORD


def get_perf_from_es(es, build_id):
    """ Get the perf from elastic search for the specified build id. """
    query = {
        "query": {
            "term": {
                "build_id.keyword": build_id
            }
        },
        "size": 100
    }
    perf_res = es.search(index="pybuda-perf-ci", body=query)
    hits = perf_res['hits']['hits']
    return [{
        "build_id": h["_source"]["build_id"],
        "model": h["_source"]["args"]['model'],
        "config": h["_source"]["args"]['config'],
        "dataformat": h["_source"]["args"]['dataformat'],
        "arch": h["_source"]["args"]['arch'],
        "samples_per_sec": h["_source"]["samples_per_sec"],
        "index": "_".join([h["_source"]["args"]['model'], h["_source"]["args"]['config'], h["_source"]["args"]['arch'], h["_source"]["args"]['dataformat']])
    } for h in hits]


def compare_perf(build_ids: list):
    """ Compare the perf for the specified build ids. """
    es = Elasticsearch([ES_ENDPOINT], http_auth=(ES_USERNAME, ES_PASSWORD))
    print("\nGetting perf from elastic search\n")

    # Load perf data from elastic search
    data = []
    for build_id in build_ids:
        es_res = get_perf_from_es(es, build_id)
        data.extend(es_res)
        print(f"Got perf for build id: {build_id} with {len(es_res)} records")

    # Build the table
    keys = set()
    build_ids = set()
    for d in data:
        keys.add(d["index"])
        build_ids.add(d["build_id"])
    build_ids = list(build_ids)

    table = {}
    for d in data:
        if not d['index'] in table:
            row = {"arch": d["arch"], "config": d["config"], "dataformat": d["dataformat"], "model": d["model"]}
            for b in build_ids:
                row[b] = 0.0
            table[d['index']] = row
        score = d['samples_per_sec']
        if not isinstance(score, float):
            score = 0
        table[d['index']][d['build_id']] = score

    # Add perf diff - assume we are comparing only two build ids
    for k, v in table.items():
        perf_dif = v[build_ids[1]] - v[build_ids[0]]
        max_val = max(v[build_ids[1]], v[build_ids[0]])
        table[k]["diff"] = f"{perf_dif:.2f}"
        if max_val > 0:
            table[k]["diff_pct"] = f"{ 100 * perf_dif / max_val :.2f}%"
        else:
            table[k]["diff_pct"] = "N/A"

    # Print the table
    rows = [r for r in table.values()]
    print("\nPerf comparison\n")
    print(tabulate.tabulate(rows, headers="keys"))


def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_perf.py <build_id1> <build_id2>")
        sys.exit(1)

    build_ids = sys.argv[1:]

    # correct the build ids prefix
    prefix = "gitlab-pipeline-"
    build_ids = [(x if x.startswith(prefix) else prefix + str(x)) for x in build_ids]

    # compare
    compare_perf(build_ids)

if __name__ == "__main__":
    main()


def test_compare_perf():
    compare_perf(["gitlab-pipeline-479274", "gitlab-pipeline-479323"])
