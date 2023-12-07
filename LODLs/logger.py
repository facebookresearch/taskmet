import os
import matplotlib.pyplot as plt
import csv

# Logger object to store the train and validation metrics to be used in workspcae object during training and also enable plotting at the of training
class Logger:
    def __init__(
        self, work_dir, filename, print_freq=10, save_freq=1, save_fig_freq=50
    ):
        self.work_dir = work_dir
        self.filename = filename
        self.print_freq = print_freq
        self.train_metrics = {}
        self.val_metrics = {}
        self.log_file = open(os.path.join(self.work_dir, self.filename), "w")
        self.csv_writer = csv.writer(
            self.log_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        self.save_freq = save_freq
        self.save_fig_freq = save_fig_freq

    def log(self, metrices={}, iter=0, partition="Train"):
        # print the metrices in neat format
        if iter % self.print_freq == 0:
            self.print_metrics(metrices, iter, partition)

        # write the metrices to csv file
        if iter == 0 and partition == "Train":
            self.csv_writer.writerow(["partition", "iter"] + list(metrices.keys()))
        if iter % self.save_freq == 0:
            self.csv_writer.writerow([partition, iter] + list(metrices.values()))
            self.log_file.flush()

        # append the existing metrics and add new key if metric is not present
        for metric in metrices.keys():
            if partition == "Train":
                if metric not in self.train_metrics.keys():
                    self.train_metrics[metric] = {}
                self.train_metrics[metric][iter] = metrices[metric]
            else:
                if metric not in self.val_metrics.keys():
                    self.val_metrics[metric] = {}
                self.val_metrics[metric][iter] = metrices[metric]

        if iter % self.save_fig_freq == 0 and partition == "Val":
            self.plot()

    def print_metrics(self, metrics={}, iter=0, parition="Train"):
        print(f"{parition:<7} Iter: {iter}", end=" ")
        for metric in metrics.keys():
            print(f"{metric}: {metrics[metric]:.2e}", end=" ")
        print()

    # plot all the metrices and save them in work_dir
    def plot(self):
        # create individual plots for each metric
        for metric in self.train_metrics.keys():
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            ax.plot(
                list(self.train_metrics[metric].keys()),
                list(self.train_metrics[metric].values()),
                label=f"train_{metric}",
            )
            ax.plot(
                list(self.val_metrics[metric].keys()),
                list(self.val_metrics[metric].values()),
                label=f"val_{metric}",
            )
            ax.set_xlabel("Outer iters")
            ax.set_ylabel(metric)
            ax.set_yscale("log")
            ax.legend()
            fig.savefig(os.path.join(self.work_dir, f"{metric}.png"))
            ax.cla()
            plt.close(fig)

    def close(self):
        self.plot()
        plt.close("all")
        self.log_file.close()
