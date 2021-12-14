from pyspark.sql import SparkSession

file = open('text')
text = file.read().split(' ')
spark = SparkSession.builder.getOrCreate()
mywords_terry = spark.sparkContext.parallelize(text, 4)
mywords_terry.setName("mywords_terry_niagara")
print('partition name is ' + str(mywords_terry.name()))
print('initial partition count: ' + str(mywords_terry.getNumPartitions()))
print('unique words: ' + str(mywords_terry.distinct().count()))

def startsWithf(individual):
	return individual.startswith('F')
print('words start with F')
print(mywords_terry.filter(lambda word: startsWithf(word)).collect())
print('number of words start with F')
print(mywords_terry.filter(lambda word: startsWithf(word)).count())

# longest word
def longest_reduce(leftword, rightword):
	if len(leftword) > len(rightword):
		return leftword
	else:
		return rightword
print('the longest word is ' + str(mywords_terry.reduce(longest_reduce).split('\n')[0]))

# shortest word
def shortest_reduce(leftword, rightword):
	if len(leftword) < len(rightword):
		return leftword
	else:
		return rightword
mywords_terry = mywords_terry.filter(lambda word: word != '')
print('the shortest word is ' + str(mywords_terry.reduce(shortest_reduce)))