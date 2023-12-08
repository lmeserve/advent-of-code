import polars as pl

with open('./2023/data/day2.txt') as f:
  data = f.readlines()

df_raw = pl.DataFrame({"x": data})


def unpack_game(df):
  return(
    df
    .with_columns([
      pl.col('x')
      .str.split_exact(":", 2)
      .struct.rename_fields(["game", "results"])
      .alias("fields")
    ])
    .unnest('fields')
    .drop('x')
    .with_columns(pl.col('game').str.replace("Game ", ""))
    .with_columns(pl.col('game').cast(pl.Int8))
  )

def unpack_rounds(df):
  return (
    df
    .with_columns(
      pl.col('results')
      .str.split(";")
    )
    .with_columns(pl.col('results').list.lengths().alias('rounds'))
    .with_columns(pl.col('rounds').apply(lambda x: range(1, x + 1)))
    .explode('results', 'rounds')
) 

def unpack_results(df):
  return (
    df
    .with_columns(
      pl.col('results')
      .str.split(",")
    )
    .explode('results')
    .with_columns(pl.col("results").str.strip())

    .with_columns(
      pl.col('results')
      .str.split_exact(" ", 2)
      .struct.rename_fields(["n", "colour"])
      .alias("fields")
    )
    .unnest("fields")
    .drop('results')
    .with_columns(pl.col('n').cast(pl.Int8))
  )

df = df_raw.pipe(unpack_game).pipe(unpack_rounds).pipe(unpack_results)

df_impossible = (
  df.filter(
    ((pl.col('n') > 12) & (pl.col('colour') == 'red')) |
    ((pl.col('n') > 13) & (pl.col('colour') == 'green')) |
    ((pl.col('n') > 14) & (pl.col('colour') == 'blue'))
  )
)

df_possible = df.join(df_impossible, on='game', how='anti')

sum(
  df_possible
  .select("game")
  .unique()
  .get_column('game')
  .to_list()
)


#### Part 2 ####
with pl.Config(tbl_rows=20):
    print(df.filter(pl.col('game') == 1))


df_max_colours = df.groupby('game', 'colour').agg(pl.col('n').max()).sort('game')

df_powers = df_max_colours.groupby('game').agg(pl.col('n').product())

sum(df_powers.get_column('n').to_list())
