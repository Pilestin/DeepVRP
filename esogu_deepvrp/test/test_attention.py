import sys
import io
sys.stdin = io.StringIO("2\n")

from main import main

if __name__ == "__main__":
    vrp_problem, dl_data, results = main()
