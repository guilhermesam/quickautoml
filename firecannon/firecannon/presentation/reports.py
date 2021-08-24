from pandas import DataFrame
from numpy import mean, std
from matplotlib import pyplot as plt
import re
import csv


class Report:
    def get_columns_names(self, data):
        metric_names = list(data.values())[0].keys()
        columns = []
        [columns.extend([f'mean_{metric}', f'std_{metric}']) for metric in metric_names]
        return columns

    def get_metrics_data(self, data):
        structured_data = []
        for model_metrics in data.values():
            metrics = []
            for metric in list(model_metrics.values()):
                metrics.extend([mean(metric), std(metric)])
            structured_data.append(metrics)
        return structured_data


class DataframeReport(Report):
    def make_report(self, data: dict) -> DataFrame:
        columns = self.get_columns_names(data)
        scores_df = DataFrame(columns=columns)

        for model_metrics in data.values():
            metrics = []
            for metric in list(model_metrics.values()):
                metrics.extend([mean(metric), std(metric)])
            scores_df.loc[len(scores_df)] = metrics

        scores_df.index = data.keys()
        print(scores_df)
        return scores_df


class BarplotReport(Report):
    def make_report(self, data: dict) -> None:
        results = DataFrame(
            data=self.get_metrics_data(data),
            columns=self.get_columns_names(data)
        )
        mean_regexp = re.compile('mean[_]')
        mean_columns = [col for col in results.columns if bool(re.match(mean_regexp, col))]

        for metric in mean_columns:
            fig, _ = plt.subplots(figsize=(10, 8))
            mean_metric = results.loc[:, metric]
            plt.bar(mean_metric.index, mean_metric.values)
            plt.grid(linestyle='dotted')
            plt.title(mean_metric.name)
            fig.savefig(f'{mean_metric.name}.png')


class CsvReport(Report):
    def make_report(self, data: dict) -> None:
        with open('evaluation_results.csv', 'w', encoding='UTF8', newline='') as file:
            writer = csv.writer(file)
            columns = self.get_columns_names(data)
            writer.writerow(columns)

            for row in self.get_metrics_data(data):
                writer.writerow(row)

        file.close()
