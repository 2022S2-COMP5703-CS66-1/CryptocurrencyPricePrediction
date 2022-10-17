from models.informer.utils import Trainer
# !!!! Please make sure the import path is correct

trainer = Trainer("data/hourlyzscore.csv")
# please also make sure the path to data file is correct
# For more hyperparameters, please see the source code

trainer.train()
# the performance data, model file, hyperparameters will be dumped into a folded named by time.
