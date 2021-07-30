from pandas import DataFrame
from numpy import mean, std
from matplotlib import pyplot as plt
import re


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
        mean_regexp = re.compile('mean[_]')
        mean_columns = [col for col in results.columns if bool(re.match(mean_regexp, col))]

        for metric in mean_columns:
            fig, _ = plt.subplots(figsize=(10, 8))
            mean_metric = results.loc[:, metric]
            plt.bar(mean_metric.index, mean_metric.values)
            plt.grid(linestyle='dotted')
            plt.title(mean_metric.name)
            fig.savefig(f'{mean_metric.name}.png')
