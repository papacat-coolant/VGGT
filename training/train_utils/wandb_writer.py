import atexit
import logging
import uuid
import torch

import wandb

from .distributed import get_machine_local_and_dist_rank

class WandBLogger:

    def __init__(
        self, project, *args, group=None, job_type=None, name=None, notes=None, tags=None, config=None, dir=None, resume=None, **kwargs,
    ):
    self._run = None
    _, self._rank = get_machine_local_and_dist_rank()

    if self._rank == 0:
        wandb.login()
        self._run = wandb.init(
            project=project,
            group = group,
            job_type = job_type,
            name = name or str(uuid.uuid4()),
            notes = notes,
            tags = tags,
            config = config,
            dir = dir,
            resume = resume,
            *args,
            **kwargs,
        )
    atexit.register(self.close)

    @property
    def run(self):
        return self._run

    def flush(self):
        if self._run:
            self._run.log({})

    def close(self):
        if self._run:
            self._run.finish()
            self._run = None
    def watch_model(self, model, log_freq=1000):
        if not self._run:
            return
        self._run.watch(model, log_freq=log_freq) 

    def log_dict(self, payload, step):
        if not self._run:
            return

        if step is not None:
            payload["step"] = step

        self._run.log(payload)

    def log(self, name, data,step):
        if not self._run:
            return
        payload = {name:data}
        self._run.log(payload,step=step)

    def log_visuals(self,name,data,step,fps):
        if data.ndim == 3:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()

            if data.shape[0] <= 4 and data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
                data = data.transpose(1, 2, 0)

            wandb_image = wandb.Image(data, caption=name)
            payload = {name: wandb_image}

            if step is not None:
                payload['step'] = step

            self._run.log(payload)

        elif data.ndim == 5:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()

            wandb_video = wandb.Video(data, caption=name, fps=fps)
            payload = {name: wandb_video}

            if step is not None:
                payload["step"] = step

            self._run.log(payload)

               