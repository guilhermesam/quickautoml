from pandas import DataFrame
from numpy import mean, std


class DataframeReport:
    @staticmethod
    def make_report(data: dict) -> DataFrame:
        metric_names = list(data.values())[0].keys()
        columns = []
        [columns.extend([f'mean_{metric}', f'std_{metric}']) for metric in metric_names]
        scores_df = DataFrame(columns=columns)

        for model_metrics in data.values():
            metrics = []
            for metric in list(model_metrics.values()):
                metrics.extend([mean(metric), std(metric)])
            scores_df.loc[len(scores_df)] = metrics

        scores_df.index = data.keys()
        return scores_df


class BarplotReport:
    @staticmethod
    def make_report(data):
        results = DataframeReport.make_report(data)

        for model in results.index:
            print(results.loc[model])
