'''Day 5'''
import polars as pl
import numpy as np
import numba as nb
# from timeit import default_timer as timer
import time

FIXTURE = [
  'seeds: 79 14 55 13',
  '\n',
  'seed-to-soil map:',
  '50 98 2',
  '52 50 48',
  '\n',
  'soil-to-fertilizer map:',
  '0 15 37',
  '37 52 2',
  '39 0 15',
  '\n',
  'fertilizer-to-water map:',
  '49 53 8',
  '0 11 42',
  '42 0 7',
  '57 7 4',
  '\n',
  'water-to-light map:',
  '88 18 7',
  '18 25 70',
  '\n',
  'light-to-temperature map:',
  '45 77 23',
  '81 45 19',
  '68 64 13',
  '\n',
  'temperature-to-humidity map:',
  '0 69 1',
  '1 0 69',
  '\n',
  'humidity-to-location map:',
  '60 56 37',
  '56 93 4'
]

#### Functions
def unpack_data(L):
  seeds = L[:L.index('\n')]
  extra = L[L.index('\n') + 1:]

  seed_to_soil = extra[:extra.index('\n')][1:]
  extra = extra[extra.index('\n') + 1:]

  soil_to_fertilizer = extra[:extra.index('\n')][1:]
  extra = extra[extra.index('\n') + 1:]

  fertilizer_to_water = extra[:extra.index('\n')][1:]
  extra = extra[extra.index('\n') + 1:]

  water_to_light = extra[:extra.index('\n')][1:]
  extra = extra[extra.index('\n') + 1:]

  light_to_temperature = extra[:extra.index('\n')][1:]
  extra = extra[extra.index('\n') + 1:]

  temperature_to_humidity = extra[:extra.index('\n')][1:]

  humidity_to_location = extra[extra.index('\n') + 1:][1:]

  return ({
    'seeds': seeds,
    'seed_to_soil': seed_to_soil,
    'soil_to_fertilizer': soil_to_fertilizer,
    'fertilizer_to_water': fertilizer_to_water,
    'water_to_light': water_to_light,
    'light_to_temperature': light_to_temperature,
    'temperature_to_humidity': temperature_to_humidity,
    'humidity_to_location': humidity_to_location,
  })

def list_to_df(l):
  return (
    pl.DataFrame({'fields': l})
    .with_columns(  
      pl.col('fields').str.split_exact(" ", 2).struct.rename_fields(['dest_range_start', 'source_range_start', 'range_len'])
    )
    .unnest("fields")
    .with_columns(pl.col('range_len').str.strip())
    .with_columns(
      pl.col('dest_range_start').cast(pl.Int64),
      pl.col('source_range_start').cast(pl.Int64),
      pl.col('range_len').cast(pl.Int64)
    )
    .with_columns((pl.col('source_range_start') + pl.col('range_len') - pl.lit(1)).alias('source_range_end'))
    .with_columns((pl.col('dest_range_start') + pl.col('range_len') - pl.lit(1)).alias('dest_range_end'))
    .select('source_range_start', 'source_range_end', 'dest_range_start', 'dest_range_end')
  )

def range_between_cols(df, col_a, col_b):
  return (
    df
    .with_columns(
      pl.struct(col_a, col_b).alias('range')
    )
    .with_columns(
      pl.col('range').apply(lambda x: range(x[col_a], x[col_a] + x[col_b]))
    )
  )

def range_to_values(df):
  return (
    df
    .with_columns(
      pl.col('dest_range_start').cast(pl.Int64),
      pl.col('source_range_start').cast(pl.Int64),
      pl.col('range_len').cast(pl.Int64)
    )
    .pipe(range_between_cols, col_a='dest_range_start', col_b='range_len')
    .rename({'range': 'destination_values'})
    .pipe(range_between_cols, col_a='source_range_start', col_b='range_len')
    .rename({'range': 'source_values'})
    .explode('source_values', 'destination_values')
    .select('source_values', 'destination_values')
  )

def fill_col_forward(df, source_col):
  return (
    df
    .with_columns(
      pl.when(pl.col('destination_values').is_null()).then(pl.col(source_col)).otherwise(pl.col('destination_values')).alias('destination_values')
    )
  )

def expand_seeds(seeds):
  range_starts = [seeds[i] for i in range(len(seeds)) if i % 2 == 0]
  range_lens = [seeds[i] for i in range(len(seeds)) if i % 2 != 0]

  l = []
  for i, j in zip(range_starts, range_lens):
    l.append(np.arange(i, i+j).tolist())

  seeds_expanded = [item for sublist in l for item in sublist]
  
  if test_run:
    assert len(seeds_expanded) == 27

  return seeds_expanded

def operation(df, source):
  df = (
    df
    .filter( (pl.col('source_range_start') <= source) & (pl.col('source_range_end') >= source) )
    .with_columns((pl.lit(source) - pl.col('source_range_start')).alias('offset'))
    .filter(pl.col('dest_range_end') >= pl.col('offset'))
    .with_columns((pl.col('dest_range_start') + pl.col('offset')).alias('offset').alias('dest'))
  )

  if df.height == 0:
    result = source
  else:
    result = df.select('dest').item()
  
  return result

def get_location(seed):
  return(
    operation(df_humidity_to_location,
              operation(df_temperature_to_humidity,
                        operation(df_light_to_temperature, 
                                  operation(df_water_to_light,
                                            operation(df_fertilizer_to_water,
                                                      operation(df_soil_to_fertilizer,
                                                                operation(df_seed_to_soil, seed)))))))
  )

def operation_np(map_array, source):
  ## Find the entry relevant entry in the maping table
  test_less_than = np.less_equal(map_array, source)
  test_more_than = np.greater_equal(map_array, source)
  test = []
  for less, more in zip(test_less_than, test_more_than):
    test.append(all([less[0], more[1]]))

  i = test.index(True)

  ## Try to discover to destination value
  try:
    entry = map_array[i]
    source_start = entry[0]
    source_end   = entry[1]
    dest_start   = entry[2]
    dest_end     = entry[3]
    offset = source - source_start
    dest = (dest_start + offset)

    if dest > dest_end:
      result = source
    else:
      result = dest
  except Exception:
    result = source

  return result

def run_operation_np(target_array, source_array):
  return [operation_np(target_array, item) for item in source_array]

# def get_location_np(seed):
#   return(
#     run_operation_np(np_humidity_to_location,
#               run_operation_np(np_temperature_to_humidity,
#                         run_operation_np(np_light_to_temperature, 
#                                   run_operation_np(np_water_to_light,
#                                             run_operation_np(np_fertilizer_to_water,
#                                                       run_operation_np(np_soil_to_fertilizer,
#                                                                 run_operation_np(np_seed_to_soil, seed)))))))
#   )


#### Load data
test_run = True
if test_run:
  data_lists = unpack_data(FIXTURE)
else:
  with open('./2023/data/day5.txt') as f:
    data = f.readlines()

  data_lists = unpack_data(data)

seeds = [int(seed) for seed in data_lists['seeds'][0].replace('seeds: ', '').split(' ')]
df_seeds = pl.DataFrame({'seed_no': seeds})
df_seed_to_soil = list_to_df(data_lists['seed_to_soil'])
df_soil_to_fertilizer = list_to_df(data_lists['soil_to_fertilizer'])
df_fertilizer_to_water = list_to_df(data_lists['fertilizer_to_water'])
df_water_to_light = list_to_df(data_lists['water_to_light'])
df_light_to_temperature = list_to_df(data_lists['light_to_temperature'])
df_temperature_to_humidity = list_to_df(data_lists['temperature_to_humidity'])
df_humidity_to_location = list_to_df(data_lists['humidity_to_location'])

#### Part 1
soils = [operation(df_seed_to_soil, seed) for seed in seeds]
fertilizers = [operation(df_soil_to_fertilizer, soil) for soil in soils]
waters = [operation(df_fertilizer_to_water, fertilizer) for fertilizer in fertilizers]
lights = [operation(df_water_to_light, water) for water in waters]
temperatures = [operation(df_light_to_temperature, light) for light in lights]
humidities = [operation(df_temperature_to_humidity, temperature) for temperature in temperatures]
location = [operation(df_humidity_to_location, humidity) for humidity in humidities]

min(location)

#### Part 2
# seeds_expanded = expand_seeds(seeds)
# len(seeds_expanded)

np_seed_to_soil = df_seed_to_soil.to_numpy()
np_soil_to_fertilizer = df_soil_to_fertilizer.to_numpy()
np_fertilizer_to_water = df_fertilizer_to_water.to_numpy()
np_water_to_light = df_water_to_light.to_numpy()
np_light_to_temperature = df_light_to_temperature.to_numpy()
np_temperature_to_humidity = df_temperature_to_humidity.to_numpy()
np_humidity_to_location = df_humidity_to_location.to_numpy()



range_starts = np.array([seeds[i] for i in range(len(seeds)) if i % 2 == 0])
range_lens = np.array([seeds[i] for i in range(len(seeds)) if i % 2 != 0])
range_ends = np.add(range_starts, range_lens)
seed_ranges = [[start, end] for start, end in zip(range_starts, range_ends)]
input_ranges = [[start, n] for start, n in zip(range_starts, range_lens)]


l = []
for seed_start, n_seeds in zip(range_starts, range_lens):
  for item in np_seed_to_soil:
    source_range_start = item[0]
    source_range_end   = item[1]
    dest_range_start   = item[2]
    dest_range_end     = item[3]

    seed_end = seed_start + n_seeds - 1
    start_test = (seed_start >= source_range_start)
    end_test = ( (seed_end) <= source_range_end )
    
    print(seed_start, n_seeds, seed_end, item, start_test, end_test)

    if all([start_test, end_test]):

      offset = seed_start - source_range_start

      dest_start = dest_range_start + offset
      dest_end = dest_start + n_seeds

      l.append([dest_start, dest_end])

  else:
    pass


def operation_range(input_ranges, map_array):
  l = []
  for input in input_ranges:
    for item in map_array:
      input_start = input[0]
      input_n = input[1]
      
      source_range_start = item[0]
      source_range_end   = item[1]
      dest_range_start   = item[2]
      dest_range_end     = item[3]

      input_end = input_start + input_n - 1
      start_test = (input_start >= source_range_start)
      end_test = ( (input_end) <= source_range_end )
      
      offset = input_start - source_range_start

      dest_start = dest_range_start + offset
      dest_end = dest_start + input_n

      if all([start_test, end_test]):
        l.append([dest_start, dest_end])
      # else:
      #   pass

      print(
        input, input_start, input_n, input_end,
        item,
        start_test, end_test,
        dest_start, dest_end
      )

    if len(l) == 0:
      l.append([input_start, input_n])
  
  return l

soil_ranges = operation_range(input_ranges, np_seed_to_soil)
fertilizer_ranges = operation_range(soil_ranges, np_soil_to_fertilizer)


pl.DataFrame(seeds)

source = seeds_expanded[0]
map_array = np_seed_to_soil

soil = np.array([operation_np(np_seed_to_soil, item) for item in seeds_expanded])
fertilizer = np.array([operation_np(np_soil_to_fertilizer, item) for item in soils])
water = np.array([operation_np(np_fertilizer_to_water, item) for item in fertilizers])
light = np.array([operation_np(np_water_to_light, item) for item in water])
temperature = np.array([operation_np(np_light_to_temperature, item) for item in light])
humidity = np.array([operation_np(np_temperature_to_humidity, item) for item in temperature])
location = np.array([operation_np(np_humidity_to_location, item) for item in humidity])
