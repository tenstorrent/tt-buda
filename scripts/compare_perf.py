# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import math
from elasticsearch import Elasticsearch
import pandas as pd

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

    df = pd.DataFrame.from_records(data)
    ids = df['build_id'].unique()

    # pivot table on build_id
    pivot_table = df.pivot_table(index=['index', 'arch', 'config', 'dataformat', 'model'], columns='build_id', values='samples_per_sec', aggfunc='max')
    df = pd.DataFrame(pivot_table.to_records())

    # add pct_diff column
    df.drop(columns=['index'], inplace=True)
    df['pct_diff'] = ((df[ids[1]] - df[ids[0]]) / df[ids[0]]) * 100

    # format the pct_diff column
    def format_value(value):
        if not math.isnan(value):
            if value > 1:
                return f'\x1b[32m{value:.2f}%\x1b[0m'  # ANSI code for green color
            elif value < -1:
                return f'\x1b[31m{value:.2f}%\x1b[0m'  # ANSI code for red color
            else:
                return f'\x1b[37m{value:.2f}%\x1b[0m'  # ANSI code for red color
        else:
            return 'N/A'

    df['pct_diff'] = df['pct_diff'].apply(format_value)
    df = df.round(2)
    print(df)


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
