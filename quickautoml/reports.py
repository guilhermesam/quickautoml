from csv import writer
from json import dump
from typing import Any, Dict

from pandas import Series, DataFrame
from numpy import mean, std
from matplotlib import pyplot as plt


class Report:
    def make_report(self, data: Dict[Any, float]):
        raise NotImplementedError()

    @staticmethod
    def get_columns_names(data: Dict[Any, float]):
        metric_names = list(data.values())[0].keys()
        columns = []
        [columns.extend([f'mean_{metric}', f'std_{metric}']) for metric in metric_names]
        return columns

    @staticmethod
    def get_metrics_data(data):
        structured_data = []
        for model_metrics in data.values():
            metrics = []
            for metric in list(model_metrics.values()):
                metrics.extend([mean(metric), std(metric)])
            structured_data.append(metrics)
        return structured_data


class JsonReport(Report):
    def make_report(self, data: Dict[Any, float]):
        print(data)
        with open('models.json', 'w') as models_output:
            processed_dict = {key.__class__.__name__: value for key, value in data.items()}
            dump(processed_dict, models_output)


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
        return scores_df


class BarplotReport(Report):
    def make_report(self, data: Dict[Any, float]) -> None:
        results = Series(
            data=data.values(),
            index=data.keys()
        )

        fig, _ = plt.subplots(figsize=(10, 8))
        plt.bar([x.__class__.__name__ for x in results.index], results.values)
        plt.grid(linestyle='dotted')
        plt.title('Models Comparison')
        fig.savefig('models.png')


class CsvReport(Report):
    def make_report(self, data: dict) -> None:
        with open('evaluation_results.csv', 'w', encoding='UTF8', newline='') as file:
            csv_writer = writer(file)
            columns = self.get_columns_names(data)
            csv_writer.writerow(columns)

            for row in self.get_metrics_data(data):
                csv_writer.writerow(row)

        file.close()
