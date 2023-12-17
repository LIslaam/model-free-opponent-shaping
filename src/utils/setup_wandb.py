import wandb

def setup_wandb(args):
    kwargs = {'project': "M-FOS", 'config': args, 'reinit': True,
              'settings': wandb.Settings(_disable_stats=True)}
    run = wandb.init(**kwargs)
    #wandb.save('*.txt')
    #run.save()
    return run