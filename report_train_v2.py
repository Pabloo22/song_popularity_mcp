from pandas_profiling import ProfileReport

from utils import load_data


def main():
    train, _ = load_data(split=False, version=2)
    profile = ProfileReport(train, title='train_v2 report', html={'style': {'full_width': True}})
    profile.to_file('reports/train_v2_report.html')


if __name__ == '__main__':
    main()
