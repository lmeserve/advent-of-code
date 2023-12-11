import re
import polars as pl

DIGITS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

FIXTURE = [
  "467..114..",
  "...*......",
  "..35..633.",
  "......#...",
  "617*......",
  ".....+.58.",
  "..592.....",
  "......755.",
  "...$.*....",
  ".664.598.."
]

#### Functions
def scan(grid, idx, i, j, dim):
  results = []

  postitions = [
    [i, j-1],
    [i, j+1],
    [i-1, j],
    [i-1, j-1],
    [i-1, j+1],
    [i+1, j],
    [i+1, j-1],
    [i+1, j+1]
  ]

  for position in postitions:
    x = position[0]
    y = position[1]

    if (x < 0) or (x > dim):
      pass
    elif (y < 0) or (y > dim):
      pass
    else:
      value = grid[x][y]

      results.append({'idx': idx, 'row': x, 'column': y, 'value': value})
  
  return results


#### Grid
grid = [list(item) for item in data]
  # digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
  # set_char = set([column for row in grid for column in row])
  # special_chars = [item for item in set_char if item not in digits + ['.']]

grid_positions = []
for x, rows in enumerate(grid):
  for y, column in enumerate(rows):
    grid_positions.append({'value': column, 'row': x, 'column': y})

df_grid = pl.from_dicts(grid_positions)


#### Get Numbers
numbers = []
for n, row in enumerate(data):
  for each in re.finditer('[0-9]+', row):
    numbers.append({'row': n, 'match': each.group(), 'start': each.start(), 'end': each.end()})
 
df_numbers = (
  pl.from_dicts(numbers)
  .with_row_count(name='number_id')
  .with_columns(
    pl.struct('start', 'end').alias('column')
  )
  .with_columns(
    pl.col('column').apply(lambda x: range(x['start'], x['end']))
  )
  .explode('column')
  .with_columns(pl.col('match').cast(pl.Int64).alias('number'))
  .select('number_id', 'number', 'row', 'column')
  .join(df_grid, on=['row', 'column'], how='inner')
)


#### Part 1
special_chars = (
  df_grid
  .filter(~pl.col('value').is_in(['.'] + DIGITS))
  .select('value')
  .unique()
  .get_column('value')
  .to_list()
)

num_elements = df_numbers.to_dicts()

results = []
for num in num_elements:
  results.append(scan(grid, num['number_id'], num['row'], num['column'], 139))

df_scan = pl.from_dicts([obs for res_set in results for obs in res_set])

df_matches = (
  df_scan
  .rename({'idx': 'number_id'})
  .with_columns(pl.col('number_id').cast(pl.UInt32))
  .filter(pl.col('value').is_in(special_chars))
  .select('number_id')
  .unique(maintain_order=True)
)

(
  df_matches
  .join(df_numbers, on=['number_id'], how='inner')
  .select('number_id', 'number')
  .unique(maintain_order=True)
  .sum()
)


#### Part 2
stars = df_grid.filter(pl.col('value') == '*').to_dicts()

results = []
for n, star in enumerate(stars):
  results.append(scan(grid, n, star['row'], star['column'], 139))

df_scan = pl.from_dicts([obs for res_set in results for obs in res_set])

df_matches = (
  df_scan
  .rename({'idx': 'star_id'})
  .filter(pl.col('value').is_in(DIGITS))
  .join(df_numbers, on=['row', 'column', 'value'], how='inner')
  .select('star_id', 'number_id', 'number')
  .unique()
  .sort('star_id')
  .with_columns(pl.col('number_id').count().over('star_id').alias('count'))
  .filter(pl.col('count') > 1)
)

df_matches.groupby('star_id').agg(pl.col('number').product()).sum()
