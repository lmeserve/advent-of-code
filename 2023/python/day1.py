import re

coords = [
  "1abc2",
  "pqr3stu8vwx",
  "a1b2c3d4e5f",
  "treb7uchet"
]

with open('./2023/data/day1.txt') as f:
  data = f.readlines()

assert len(data) == 1000

text = [item.strip() for item in data]

assert len(text) == 1000

#### part 1 ####
result = ["".join([re.sub("[a-z]", "", item)[0], re.sub("[a-z]", "", item)[-1]]) for item in text]

sum([int(item) for item in result])

#### Part 2 ####
def part_two(text):
  def find_overlaps(s, pattern):
    l = []
    for m in re.finditer(pattern, s):
      l.append(m.group(1))

    return l

  def word2number(item):  
    try:
      num = str(number_dictionary[item])
    except:
      num = item
    return num
  
  number_dictionary = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
  pattern = "(?=(" + "|".join(number_dictionary.keys()) + "|" +  "|".join([str(item) for item in number_dictionary.values()]) + "))"
  # num_list = [re.findall(pattern, item) for item in text]
  num_list = [find_overlaps(item, pattern) for item in text]
  digit_list = [[word2number(item) for item in list] for list in num_list]
  pairs = [[item[0], item[-1]] for item in digit_list]
  numbers = ["".join(pair) for pair in pairs]
  integers = [int(item) for item in numbers]

  return integers

re.findall("1|2|3|(?=(one))|(?=(eight))", "z234oneight")

pattern = "(?=(one|two|three|four|five|six|seven|eight|nine|1|2|3|4|5|6|7|8|9))"





ss = "AAAA"
for m in re.finditer('(?=(AA))', ss):
    print(m.start(), m.end(), m.group(1))

fixture = [
  "two1nine",
  "eightwothree",
  "abcone2threexyz",
  "xtwone3four",
  "4nineeightseven2",
  "zoneight234",
  "7pqrstsixteen"
]

expected = 281

assert sum(part_two(fixture)) == expected

assert sum(part_two(text)) != 55648
