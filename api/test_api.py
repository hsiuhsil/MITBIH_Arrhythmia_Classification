import argparse
import numpy as np
import pandas as pd
import requests

def main():
    parser = argparse.ArgumentParser(description="Test the ECG prediction API")
    parser.add_argument("--csv", type=str, help="Path to CSV file containing beats (N x 260)")
    parser.add_argument("--num_beats", type=int, default=5, help="Number of random beats to generate if no CSV provided")
    parser.add_argument("--url_json", type=str, default="http://127.0.0.1:8000/predict_json", help="JSON API endpoint URL")
    parser.add_argument("--url_csv", type=str, default="http://127.0.0.1:8000/predict_csv", help="CSV API endpoint URL")
    args = parser.parse_args()

    # --- JSON input ---
    if args.csv:
        beats = pd.read_csv(args.csv, header=None).to_numpy().tolist()
    else:
        beats = np.random.randn(args.num_beats, 260).tolist()

    print("Testing JSON endpoint...")
    response = requests.post(args.url_json, json={"data": beats})  # key must match Pydantic model
    if response.status_code == 200:
        print("JSON API Response:")
        print(response.json())
    else:
        print(f"JSON request failed with status code {response.status_code}")
        print(response.text)

    # --- CSV input (if CSV file is provided) ---
    if args.csv:
        print("\nTesting CSV endpoint...")
        with open(args.csv, "rb") as f:
            files = {"file": (args.csv, f, "text/csv")}
            response = requests.post(args.url_csv, files=files)
        if response.status_code == 200:
            print("CSV API Response:")
            print(response.json())
        else:
            print(f"CSV request failed with status code {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    main()
