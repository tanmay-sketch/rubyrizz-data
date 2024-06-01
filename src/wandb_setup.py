import wandb

def initialize_wandb(project_name, config):
    wandb.init(project=project_name)
    wandb.config.update(config)
    
def log_metrics(metrics, step):
    wandb.log(metrics, step=step)
    
def finish_wandb():
    wandb.finish()
