# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import logging
import pandas as pd
from databricks import sql

log = logging.getLogger(__name__)

"""
Sample tables are created here

  - E2-dogfood: https://e2-dogfood.staging.cloud.databricks.com/?o=6051921418418893#notebook/3642707736157009/command/551761898400018
  - Data Force One: https://dbc-559ffd80-2bfc.cloud.databricks.com/?o=7395834863327820#notebook/2500382962301597/command/2500382962301599

The script can be called as:

  - python scripts/data_prep/convert_delta_to_json.py  --delta_table_name 'main.streaming.dummy_table' --json_output_path /tmp/delta2json2 --debug False --http_path 'sql/protocolv1/o/7395834863327820/1116-234530-6seh113n'

  - python scripts/data_prep/convert_delta_to_json.py  --delta_table_name 'main.streaming.dummy_table' --json_output_path /tmp/delta2json2 --debug False --http_path /sql/1.0/warehouses/7e083095329f3ca5 --DATABRICKS_HOST e2-dogfood.staging.cloud.databricks.com --DATABRICKS_TOKEN dapi18a0a6fa53b5fb1afbf1101c93eee31f

"""

def stream_delta_to_json(connection, tablename, json_output_folder, key = 'name', batch_size=3):

    cursor = connection.cursor()
    cursor.execute(f"USE CATALOG main;")
    cursor.execute(f"USE SCHEMA streaming;")
    cursor.execute(f"SELECT COUNT(*) FROM {tablename}")
    ans = cursor.fetchall()

    total_rows = [ row.asDict() for row in ans ][0].popitem()[1]
    print('total_rows = ', total_rows)

    cursor.execute(f"SHOW COLUMNS IN {tablename}")
    ans = cursor.fetchall()

    order_by = [ row.asDict() for row in ans ][0].popitem()[1]
    print('order by column ', order_by)

    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        query = f"""
        WITH NumberedRows AS (
            SELECT
                *,
                ROW_NUMBER() OVER (ORDER BY {key}) AS rn
            FROM
                {tablename}
        )
        SELECT *
        FROM NumberedRows
        WHERE rn BETWEEN {start+1} AND {end}"""
        cursor.execute(query)
        ans = cursor.fetchall()

        result = [ row.asDict() for row in ans ]
        print(result)
        df = pd.DataFrame.from_dict(result)
        df.to_json(os.path.join(args.json_output_path, f'shard_{start+1}_{end}.json'))

    cursor.close()
    connection.close()


if __name__ == "__main__":
    print(f"Start .... Convert delta to json")
    parser = argparse.ArgumentParser(description="Download delta table from UC and convert to json to save local")
    parser.add_argument("--delta_table_name", required=True, type=str, help="UC table of format <catalog>.<schema>.<table name>")
    parser.add_argument("--json_output_path", required=True, type=str, help="Local path to save the converted json")
    parser.add_argument("--DATABRICKS_HOST", required=False, type=str, help="DATABRICKS_HOST")
    parser.add_argument("--DATABRICKS_TOKEN", required=False, type=str, help="DATABRICKS_TOKEN")
    parser.add_argument("--http_path", required=True, type=str, help="http_path from either dedicated cluster or serverless sql warehouse")
    parser.add_argument("--debug", type=bool, required=False, default=False)
    args = parser.parse_args()

    server_hostname = args.DATABRICKS_HOST  if args.DATABRICKS_HOST else os.getenv("DATABRICKS_HOST")
    access_token = args.DATABRICKS_TOKEN  if args.DATABRICKS_TOKEN else os.getenv("DATABRICKS_TOKEN")
    http_path= args.http_path # "

    try:
        connection = sql.connect(
                server_hostname=server_hostname,
                http_path=http_path,
                access_token=access_token,
            )
    except Exception as e:
        raise RuntimeError("Failed to create sql connection to db workspace. Check {server_hostname} and {http_path} and access token!") from exc

    #if os.path.exists(args.json_output_path):
    #    if not os.path.isdir(args.json_output_path) or os.listdir(args.json_output_path):
    #        raise RuntimeError(f"A file or a folder {args.json_output_path} already exists and is not empty. Remove it and retry!")

    os.makedirs(args.json_output_path, exist_ok=True)
    log.info(f"Directory {args.json_output_path} created.")

    stream_delta_to_json(connection, args.delta_table_name, args.json_output_path)

    print(f"Convert delta to json is done. check {args.json_output_path}.")
    log.info(f"Convert delta to json is done. check {args.json_output_path}.")