"""Autograding script"""

import os

# test code files
assert os.path.exists("src/6b_api_server/config.json")
assert os.path.exists("src/6b_api_server/app.py")


# test run
assert os.path.exists("datalake/logs/api_client.log")
