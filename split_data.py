import click
import pandas as pd
from sklearn.model_selection import train_test_split

@click.command()
@click.argument('input_csv', type=click.Path(exists=True))
@click.option('--test-size', '-t', default=0.2, show_default=True,
              help='Proportion of the dataset to include in the validation split (float between 0 and 1).')
@click.option('--random-state', '-r', default=42, show_default=True,
              help='Random seed for reproducible splits.', type=int)
@click.option('--train-file', default='train.csv', show_default=True,
              help='Filename for the training set (written in current directory).')
@click.option('--val-file', default='validation.csv', show_default=True,
              help='Filename for the validation set (written in current directory).')
def split_data(input_csv: str,
               test_size: float,
               random_state: int,
               train_file: str,
               val_file: str):
    """
    Reads INPUT_CSV, splits into train/validation, and writes two files into the current directory.
    """
    # Load data
    try:
        df = pd.read_csv(input_csv)
        click.echo(f"Loaded {len(df)} rows from {input_csv}.")

        # Split
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True
        )
        click.echo(f"Split into {len(train_df)} train and {len(val_df)} validation rows.")

        # Write out to current directory
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)

        click.echo(f"Wrote training set to {train_file}")
        click.echo(f"Wrote validation set to {val_file}")
    except Exception as e:
        click.echo(f"ERROR: Split program failed: {e}", err=True)

if __name__ == '__main__':
    split_data()

