import torch
from training import SAC
from pytorch_lightning import Trainer

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpus = torch.cuda.device_count()

model = SAC(
    "FetchReachDense-v1", 
    lr=1e-3, alpha=0.002, 
    tau=0.1)

trainer = Trainer(
    gpus = num_gpus,
    max_epochs = 6000,
    log_every_n_steps = 1
)

trainer.fit(model)