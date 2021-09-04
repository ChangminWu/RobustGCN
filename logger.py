# from https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/arxiv/logger.py

import torch


class Logger(object):
    def __init__(self, num_runs, num_splits=1, info=None, log_handler=None):
        self.info = info
        self.log_handler = log_handler
        self.results = [[[] for _ in range(num_runs)] for _ in range(num_splits)]

    def add_result(self, run, result, split=0):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results[split]) and split < len(self.results)
        self.results[split][run].append(result)

    def print_statistics(self, run=None, split=None):
        if split is not None:

            if run is not None:
                result = 100 * torch.tensor(self.results[split][run])
                argmax = result[:, 1].argmax().item()
                self.log_handler.info(f'Split {split + 1:02d} Run {run + 1:02d}:')
                self.log_handler.info(f'Highest Train: {result[:, 0].max():.2f}')
                self.log_handler.info(f'Highest Valid: {result[:, 1].max():.2f}')
                self.log_handler.info(f'  Final Train: {result[argmax, 0]:.2f}')
                self.log_handler.info(f'   Final Test: {result[argmax, 2]:.2f}')

            else:
                result = 100 * torch.tensor(self.results[split])
                best_results = []
                for r in result:
                    argmax = r[:, 1].argmax().item()
                    train1 = r[:, 0].max().item()
                    valid = r[:, 1].max().item()
                    train2 = r[argmax, 0].item()
                    test = r[argmax, 2].item()
                    best_results.append((train1, valid, train2, test))

                best_result = torch.tensor(best_results)

                means = best_result.mean(dim=0)
                stds = best_result.std(dim=0)

                self.log_handler.info(f'Split {split + 1:02d} All Runs:')
                self.log_handler.info(f'Highest Train: {means[0]:.2f} ± {stds[0]:.2f}')
                self.log_handler.info(f'Highest Valid: {means[1]:.2f} ± {stds[1]:.2f}')
                self.log_handler.info(f'  Final Train: {means[2]:.2f} ± {stds[2]:.2f}')
                self.log_handler.info(f'   Final Test: {means[3]:.2f} ± {stds[3]:.2f}')

        else:
            assert run is None
            best_results = []
            for res in self.results:
                result = 100 * torch.tensor(res)
                for r in result:
                    argmax = r[:, 1].argmax().item()
                    train1 = r[:, 0].max().item()
                    valid = r[:, 1].max().item()
                    train2 = r[argmax, 0].item()
                    test = r[argmax, 2].item()
                    best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            means = best_result.mean(dim=0)
            stds = best_result.std(dim=0)

            self.log_handler.info(f'All Splits All Runs:')
            self.log_handler.info(f'Highest Train: {means[0]:.2f} ± {stds[0]:.2f}')
            self.log_handler.info(f'Highest Valid: {means[1]:.2f} ± {stds[1]:.2f}')
            self.log_handler.info(f'  Final Train: {means[2]:.2f} ± {stds[2]:.2f}')
            self.log_handler.info(f'   Final Test: {means[3]:.2f} ± {stds[3]:.2f}')