class ImprovedEarlyStopping:
    """基于滑动窗口综合分数的早停，与 best model 选择使用同一 score。"""

    def __init__(self, patience=50, window_size=5, min_delta=1e-4,
                 acc_weight=0.4, pr_auc_weight=0.3, f1_weight=0.3):
        self.patience = patience
        self.window_size = window_size
        self.min_delta = min_delta
        self.acc_weight = acc_weight
        self.pr_auc_weight = pr_auc_weight
        self.f1_weight = f1_weight

        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.val_acc_history = []
        self.val_pr_auc_history = []
        self.val_f1_history = []

    def compute_score(self, val_acc, val_pr_auc, val_f1):
        self.val_acc_history.append(val_acc)
        self.val_pr_auc_history.append(val_pr_auc)
        self.val_f1_history.append(val_f1)

        if len(self.val_acc_history) > self.window_size:
            self.val_acc_history.pop(0)
            self.val_pr_auc_history.pop(0)
            self.val_f1_history.pop(0)

        avg_acc = sum(self.val_acc_history) / len(self.val_acc_history)
        avg_pr_auc = sum(self.val_pr_auc_history) / len(self.val_pr_auc_history)
        avg_f1 = sum(self.val_f1_history) / len(self.val_f1_history)

        return (
            self.acc_weight * avg_acc
            + self.pr_auc_weight * avg_pr_auc
            + self.f1_weight * avg_f1
        )

    def step(self, val_acc, val_pr_auc, val_f1, epoch):
        """返回 (should_stop, is_best, score)"""
        score = self.compute_score(val_acc, val_pr_auc, val_f1)
        is_best = False

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            is_best = True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop, is_best, score

    def __call__(self, val_acc, val_pr_auc, val_f1, epoch=0):
        return self.step(val_acc, val_pr_auc, val_f1, epoch)
