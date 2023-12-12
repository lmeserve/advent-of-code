''' Day 4'''
import polars as pl

FIXTURE = [
  "Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53",
  "Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19",
  "Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1",
  "Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83",
  "Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36",
  "Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11"
]

data = FIXTURE

with open('./2023/data/day4.txt') as f:
  data = f.readlines()

df = (
  pl.DataFrame({'x': data})
  .with_columns(
    pl.col('x')
    .str.split_exact(':', 2)
    .struct.rename_fields(['card', 'numbers'])
    .alias("fields")
  )
  .unnest("fields")
  
  .with_columns(
    pl.col('numbers')
    .str.split_exact('|', 2)
    .struct.rename_fields(['winning', 'held'])
    .alias("fields")
  )
  .unnest("fields")
  .drop('x', 'numbers')

  .with_columns(pl.col('winning').str.replace_all('\s+', ' ').str.strip().str.split(' '))
  .with_columns(pl.col('held').str.replace_all('\s+', ' ').str.strip().str.split(' '))
)

#### Part 1
df_points = (
  df
  .with_columns(pl.col('winning').list.set_intersection('held').list.unique().alias('intersect'))
  .with_columns(pl.col('intersect').list.lengths().alias('n_intersect'))
  .with_columns(
    pl.when(pl.col('n_intersect') == 0).then(pl.lit(0))
    .otherwise((2 ** (pl.col('n_intersect') - 1))).alias('points')
  )
  .drop('winning', 'held')
)

df_points.sum().get_column('points')


#### Part 2

df_copies = (
  df
  .with_columns(pl.col('winning').list.set_intersection('held').list.unique().alias('intersect'))
  .with_columns(pl.col('intersect').list.lengths().alias('n_intersect'))
  .with_columns(pl.col('card').str.replace('Card\s+', '').cast(pl.Int64))
  .with_columns(pl.struct('card', 'n_intersect').alias('copies'))
  .with_columns(pl.col('copies').apply(lambda x: range(x['card'] + 1, x['card'] + 1 + x['n_intersect'])))
  .select('card', 'n_intersect', 'copies')
  # .explode('copies')
)


def expand_copies(df):
  return(
    df
    .explode('copies')
    .drop('n_intersect')
    .join(df_copies, left_on='copies', right_on='card', how='left')
    .drop('card')
    .rename({'copies': 'card'})
    .rename({'copies_right': 'copies'})
    .filter(pl.col('card').is_not_null())
  )

def test_all_nulls(df):
  non_null_rows = (
    df
    .select(pl.count())
    .item()
  )

  return non_null_rows


n_copies = test_all_nulls(df_copies)
l = [df_copies]
df_n = df_copies
while n_copies > 0:
  df_n = expand_copies(df_n)
  l.append(df_n)
  n_copies = test_all_nulls(df_n)

pl.concat(l).groupby('card', maintain_order=True).count()
