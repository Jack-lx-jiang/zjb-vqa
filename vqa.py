import click
'''
Usage: vqa.py [OPTIONS] MODEL_NAME COMMAND [ARGS]...

where MODEL_NAME is defined in model/__init__.py
and COMMAND is one of [train, test]

Example Usage:
vqa.py base_model train
vqa.py --data_dir path/to/data model test
vqa.py --batch 256 model train --epoch 200

Also See:
vqa.py --help
'''


@click.group()
@click.argument('model_name')
@click.option('--data_dir', default='dataset', help='Data path containing extracted feature.')
@click.option('--batch', default=128, help='Size of one batch.')
@click.pass_context
def cli(ctx, model_name, data_dir, batch):
    views = __import__('model')
    cur_model = getattr(views, model_name)

    global batch_size, model
    batch_size = batch
    model = cur_model(data_dir)


@cli.command()
@click.option('--epoch', default=100)
@click.pass_context
def train(ctx, epoch):
    model.train(batch_size, epoch)


@cli.command()
@click.option('--exp_name', default='latest.pkl')
@click.option('--pkl_name', default='latest.pkl')
@click.pass_context
def test(ctx, exp_name, pkl_name):
    model.test(batch_size, exp_name, pkl_name)


if __name__ == '__main__':
    cli()
