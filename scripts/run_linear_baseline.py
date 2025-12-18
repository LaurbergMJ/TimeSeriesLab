from __future__ import annotations

import argparse
import matplotlib.pyplot as plt

from ts_lab.data_io import load_ohlc_data
from ts_lab.features import make_supervised
from ts_lab.split import train_test_split_time
from ts_lab.models.linear_regression import make_linear_regression_pipeline
from ts_lab.evaluation import regression_report

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to CSV with columns: date, close")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    close = load_ohlc_data(args.csv)
    X, y = make_supervised(close)

    X_train, X_test, y_train, y_test = train_test_split_time(X, y, test_size=args.test_size)

    model = make_linear_regression_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = regression_report(y_test, y_pred)

    print("Linear Regression baseline:")

    for k, v in report.items():
        print(f" {k:>22s}: {v:.6f}")

    plt.figure()
    plt.plot(y_test.index, y_test.values, label="actual")
    plt.plot(y_test.index, y_pred, label="predicted")
    plt.title("Next-day log return: actual vs predicted")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()