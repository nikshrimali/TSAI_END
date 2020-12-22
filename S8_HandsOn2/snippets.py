# Write a function that adds 2 iterables a and b such that a is even and b is odd
def add_even_odd_list(l1:list,l2:list)-> list:
    return [a+b for a,b in zip(l1,l2) if a%2==0 and b%2!=0]

# Write a function that strips every vowel from a string provided
def strip_vowels(input_str:str)->str:

    vowels = ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U' ]
    return ''.join(list(filter(lambda x: x not in vowels, input_str)))

# write a function that acts like a ReLU function for a 1D array
def relu_list(input_list:list)->list:
    return [(lambda x: x if x >= 0 else 0)(x) for x in input_list]

# Write a function that generates Factorial of number
def factorial(n):
    if n == 0 or n ==1:
        return 1
    else:
        return n*factorial(n-1)

# Write a function that returns length of the list

def list_length(l):
    return len(l)

# Write a function that sorts list of numbers and returns top element

def biggest_no(l:list)->int:
    sorted(l)

# Write a function to print a string by repeating it n times

def print_repeatnstring(text:str, n:int)-> str:
    return text*n

# Write a function to merge two lists element wise

def merge_lists(l1:list, l2:list):
    return list(zip(l1,l2))

# Write a function to merge two lists element wise
def merge_lists(l1:list, l2:list):
    return list(zip(l1,l2))

# Write a function to append two lists

def append_lists(l1:list, l2:list)->list:
    return l1.extend(l2)

# Write a function to return reverse of a list

def reverse_list(l1:list)->list:
    return l1[::-1]

# Write a function to adds two lists element wise
def adds_listelements(l1:list, l2:list):
    return [i+j for i, j in zip(l1,l2)]

# Write a function to Subtracts two lists element wise
def sub_listelements(l1:list, l2:list):
    return [i-j for i, j in zip(l1,l2)]

# Write a function to adds two lists element wise only if numbers are even
def adds_listevenelements(l1:list, l2:list):
    return [i+j for i, j in zip(l1,l2) if i*j%2 == 0]

# Write a function to multiplies two lists element wise only if numbers are odd
def adds_listoddelements(l1:list, l2:list):
    return [i*j for i, j in zip(l1,l2) if i*j%2 == 1]

# Write a function that returns list of elements with n power to elements of list
def n_power(l1:list, power:int)->list:
    return [i**power for i in l1]


# Write a function that generates fibbonacci series
def Fibonacci(n:int)-> int:
    if n==1:
        fibonacci = 0
    elif n==2:
        fibonacci = 1
    else:
        fibonacci = Fibonacci(n-1) + Fibonacci(n-2)
    return fibonacci


# Write a function that returns sine value of the input
def sin(x:float) -> float:
    import math
    return math.sin(x)

# Write a function that returns derivative of sine value of the input
def derivative_sin(x:float)-> float:
    import math
    return math.cos(x)

# Write a function that returns tan value of the input
def tan(x:float) -> float:
    import math
    return math.tan(x)

# Write a function that returns derivative of tan value of the input
def derivative_tan(x:float)-> float:
    import math
    return (1/math.cos(x))**2


# Write a function that returns cosine value of the input
def cos(x:float) -> float:
    import math
    return math.cos(x)

# Write a function that returns cosine value of the input
def derivative_cos(x:float)-> float:
    import math
    return -(math.sin(x))


# Write a function that returns the exponential value of the input
def exp(x) -> float:
    import math
    return math.exp(x)

# Write a function that returns Gets the derivative of exponential of x
def derivative_exp(x:float) -> float:
    import math
    return math.exp(x)


# Write a function that returns log of a function
def log(x:float)->float:
    import math
    return math.log(x)

# Write a function that returns derivative of log of a function
def derivative_log(x:float)->float:
    return (1/x)


# Write a function that returns relu value of the input
def relu(x:float) -> float:
    x = 0 if x < 0 else x
    return x

# Write a function that returns derivative derivative relu value of the input
def derivative_relu(x:float) -> float:
    x = 1 if x > 0 else 0
    return x


# Write a function that returns runs a garbage collector
def clear_memory():
    import gc
    gc.collect()

# Write a function that calculates the average time taken to perform any transaction by  Function fn averaging the total time taken for transaction over Repetations
def time_it(fn, *args, repetitons= 1, **kwargs):
    import time
    total_time = []

    for _ in range(repetitons):
        start_time = time.perf_counter()
        fn(*args,**kwargs)
        end_time = time.perf_counter()
        ins_time = end_time - start_time
        total_time.append(ins_time)
    return sum(total_time)/len(total_time)


# Write a function to identify if value is present inside a dictionary or not
def check_value(d:dict, value)->bool:
    return any(v == value for v in dict.values())

# Write a function to identify to count no of instances of a value  inside a dictionary
def count_value(d:dict, value)->bool:
    return list(v == value for v in dict.values()).count(True)

# Write a function to identify if value is present inside a list or not
def check_listvalue(l:list, value)->bool:
    return value in l

# Write a function to identify if value is present inside a tuple or not
def check_tuplevalue(l:tuple, value)->bool:
    return value in l

# Write a function that returns lowercase string
def str_lowercase(s:str):
    return s.lower()

# Write a function that returns uppercase string
def str_uppercase(s:str):
    return s.upper()

# Write a function that removes all special characters
def clean_str(s):
    import re
    return re.sub('[^A-Za-z0-9]+', '', s)

# Write a function that returns a list sorted ascending
def ascending_sort(l:list):
    sorted(l, reverse=False)

# Write a function that returns a list sorted descending
def descending_sort(l:list):
    sorted(l, reverse=True)

# Write a function that returns a dictionary sorted descending by its values
def descending_dict_valuesort(d:dict):
    return {key: val for key, val in sorted(d.items(), reverse=True, key = lambda ele: ele[1])}

# Write a function that returns a dictionary sorted ascending by its values
def ascending_dict_valuesort(d:dict):
    return {key: val for key, val in sorted(d.items(), key = lambda ele: ele[1])}

# Write a function that returns a dictionary sorted descending by its keys
def descending_dict_keysort(d:dict):
    return {key: val for key, val in sorted(d.items(), reverse=True, key = lambda ele: ele[0])}

# Write a function that returns a dictionary sorted ascending by its keys
def ascending_dict_keysort(d:dict):
    return {key: val for key, val in sorted(d.items(), key = lambda ele: ele[0])}

# Write a function that returns a replace values in string with values provided
def replace_values(s:str, old, new)->str:
    s.replace(old, new)

# Write a function that joins elements of list
def join_elements(l:list)-> str:
    return (''.join(str(l)))

# Write a function that splits the elements of string
def split_elements(s:str, seperator)-> list:
    return s.split(seperator)

# Write a function that returns sum of all elements in the list
def sum_elements(l:list):
    return sum(l)

# Write a function that returns sum of all odd elements in the list
def sum_even_elements(l:list):
    return sum([i for i in l if i%2==0])

# Write a function that returns sum of all odd elements in the list
def sum_odd_elements(l:list):
    return sum([i for i in l if i%2==1])

# write a python function to count number of times a function is called 
def counter(fn):
    count = 0
    def inner(*args, **kwargs):
        nonlocal count
        count += 1
        print(f'Function {fn.__name__} was called {count} times.')
        return fn(*"args, **kwargs)    
    return inner

# write a python function to remove duplicate items from the list
def remove_duplicatesinlist(lst):
    return len(lst) == len(set(lst))

# write a python decorator function to find how much time user given function takes to execute
def timed(fn):
    from time import perf_counter
    from functools import wraps

    @wraps(fn) 
    def inner(*args, **kwargs):
        start = perf_counter()
        result = fn(*args, **kwargs)
        end = perf_counter()
        elapsed = end - start

        args_ = [str(a) for a in args]
        kwargs_ = ['{0}={1}'.format(k, v) for k, v in kwargs.items()]
        all_args = args_ + kwargs_
        args_str = ','.join(all_args) # now it is comma delimited

        print(f'{fn.__name__}({args_str}) took {elapsed} seconds')

        return result
    # inner = wraps(fn)(inner)
    return inner

# write a python program to add and print two user defined list using map
input_string = input("Enter a list element separated by space ")
list1  = input_string.split()
input_string = input("Enter a list element separated by space ")
list2  = input_string.split()
list1 = [int(i) for i in list1] 
list2 = [int(i) for i in list2] 
result = map(lambda x, y: x + y, list1, list2) 
print(list(result))

# write a python function to convert list of strings to list of integers
def stringlist_to_intlist(sList): 
  return(list(map(int, sList)))

# write a python function to map multiple lists using zip
def map_values(*args):
  return set(zip(*args))

# write a generator function in python to generate infinite square of numbers using yield
def nextSquare(): 
    i = 1;  
    # An Infinite loop to generate squares  
    while True: 
        yield i*i                 
        i += 1

# write a python generator function for generating Fibonacci Numbers 
def fib(limit): 
    # Initialize first two Fibonacci Numbers  
    a, b = 0, 1  
    # One by one yield next Fibonacci Number 
    while a < limit: 
        yield a 
        a, b = b, a + b

# write a python program which takes user input tuple and prints length of each tuple element
userInput = input("Enter a tuple:")
x = map(lambda x:len(x), tuple(x.strip() for x in userInput.split(',')))
print(list(x))

# write a python function using list comprehension to find even numbers in a list
def find_evennumbers(input_list):
  list_using_comp = [var for var in input_list if var % 2 == 0] 
  return list_using_comp

# write a python function to return dictionary of two lists using zip 
def dict_using_comp(list1, list2):
  dict_using_comp = {key:value for (key, value) in zip(list1, list2)} 
  return dict_using_comp

#Write a function to get list of profanity words from Google profanity URL
def profanitytextfile():
    url = "https://github.com/RobertJGabriel/Google-profanity-words/blob/master/list.txt"
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    textlist = []
    table = soup.find('table')
    trs = table.find_all('tr')
    for tr in trs:
        tds = tr.find_all('td')
        for td in tds:
            textlist.append(td.text)
    return textlist

#write a python program to find the biggest character in a string 
bigChar = lambda word: reduce(lambda x,y: x if ord(x) > ord(y) else y, word)

#write a python function to sort list using heapq 
def heapsort(iterable):
    from heapq import heappush, heappop
    h = []
    for value in iterable:
        heappush(h, value)
    return [heappop(h) for i in range(len(h))]

# write a python function to return first n items of the iterable as a list
def take(n, iterable):    
    import itertools
    return list(itertools.islice(iterable, n))

# write a python function to prepend a single value in front of an iterator 
def prepend(value, iterator):    
    import itertools
    return itertools.chain([value], iterator)

# write a python function to return an iterator over the last n items
def tail(n, iterable):    
    from collections import deque
    return iter(deque(iterable, maxlen=n))

# write a python function to advance the iterator n-steps ahead
def consume(iterator, n=None):
    import itertools
    from collections import deque
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(itertools.islice(iterator, n, n), None)

# write a python function to return nth item or a default value
def nth(iterable, n, default=None):
    from itertools import islice
    return next(islice(iterable, n, None), default)

# write a python function to check whether all elements are equal to each other
def all_equal(iterable):
    from itertools import groupby
    g = groupby(iterable)
    return next(g, True) and not next(g, False)

# write a python function to count how many times the predicate is true
def quantify(iterable, pred=bool):
    return sum(map(pred, iterable))

# write a python function to emulate the behavior of built-in map() function
def pad_none(iterable):
    """Returns the sequence elements and then returns None indefinitely.
    Useful for emulating the behavior of the built-in map() function.
    """
    from itertools import chain, repeat
    return chain(iterable, repeat(None))

# write a python function to return the sequence elements n times
def ncycles(iterable, n):
    from itertools import chain, repeat
    return chain.from_iterable(repeat(tuple(iterable), n))

# write a python function to return the dot product of two vectors
def dotproduct(vec1, vec2):
    return sum(map(operator.mul, vec1, vec2))

# write a python function to flatten one level of nesting
def flatten(list_of_lists):
    from itertools import chain
    return chain.from_iterable(list_of_lists)

# write a python function to repeat calls to function with specified arguments
def repeatfunc(func, times=None, *args):
    from itertools import starmap, repeat
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))

# write a python function to convert iterable to pairwise iterable
def pairwise(iterable):
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

# write a python function to collect data into fixed-length chunks or blocks
def grouper(iterable, n, fillvalue=None):
    from itertools import zip_longest
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

# write a python program to create round robin algorithm: "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
def roundrobin(*iterables):    
    from itertools import islice, cycle
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

# write a python function to use a predicate and return entries particition into false entries and true entries
def partition(pred, iterable):
    from itertools import filterfalse, tee
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)

# write a python function to return powerset of iterable
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    from itertools import chain, combinations
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

list(powerset([1,2,3]))

# write a python function to list all unique elements, preserving order
def unique_everseen(iterable, key=None):
    from itertools import filterfalse
    # unique_everseen('AAAABBBCCDAABBB') --> A B C D
    # unique_everseen('ABBCcAD', str.lower) --> A B C D
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element

# write a python function to list unique elements, preserving order remembering only the element just seen."
def unique_justseen(iterable, key=None):
    import operator
    from itertools import groupby    
    # unique_justseen('AAAABBBCCDAABBB') --> A B C D A B
    # unique_justseen('ABBCcAD', str.lower) --> A B C A D
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))

# write a python function to call a function repeatedly until an exception is raised.
def iter_except(func, exception, first=None):
    """Converts a call-until-exception interface to an iterator interface.
    Like builtins.iter(func, sentinel) but uses an exception instead
    of a sentinel to end the loop.
    Examples:
        iter_except(s.pop, KeyError)                             # non-blocking set iterator
    """
    try:
        if first is not None:
            yield first()            # For database APIs needing an initial cast to db.first()
        while True:
            yield func()
    except exception:
        pass

# write a python function to return random selection from itertools.product(*args, **kwds)
def random_product(*args, repeat=1):
    import random
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(map(random.choice, pools))

# write a python function to return random selection from itertools.permutations(iterable, r)
def random_permutation(iterable, r=None):
    import random
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))

# write a python function to random select from itertools.combinations(iterable, r)
def random_combination(iterable, r):
    import random
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)

# write a python function to perform random selection from itertools.combinations_with_replacement(iterable, r)
def random_combination_with_replacement(iterable, r):
    import random
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.choices(range(n), k=r))
    return tuple(pool[i] for i in indices)

# write a python function to locate the leftmost value exactly equal to x
def index(a, x):
    from bisect import bisect_left    
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

# write a python function to locate the rightmost value less than x 
def find_lt(a, x):
    from bisect import bisect_left  
    i = bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

# write a python function to find rightmost value less than or equal to x
def find_le(a, x):
    from bisect import bisect_right  
    i = bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError

# write a python function to find leftmost value greater than x
def find_gt(a, x):
    from bisect import bisect_right 
    i = bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

# write a python function to find leftmost item greater than or equal to x
def find_ge(a, x):
    from bisect import bisect_left 
    i = bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

# write a python function to map a numeric lookup using bisect
def grade(score, breakpoints=[60, 70, 80, 90], grades='FDCBA'):
    from bisect import bisect
    i = bisect(breakpoints, score)
    return grades[i]

# write a regex pattern in python to print all adverbs and their positions in user input text
import re
text = input("Enter a string: ")
for m in re.finditer(r"\w+ly", text):
    print('%02d-%02d: %s' % (m.start(), m.end(), m.group(0)))

# write a python function to read a CSV file and print its content
def read_csv(filename):
    import csv
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)

# write a python snippet to convert list into indexed tuple 
test_list = [4, 5, 8, 9, 10] 
list(zip(range(len(test_list)), test_list))

# write a python function to split word into chars
def split(word): 
    return [char for char in word]

# write a python function to pickle data to a file
def pickle_data(data, pickle_file):
  import pickle
  with open(pickle_file, 'wb') as f:
      pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
  return None

# write a python function to load pickle data from a file
def load_pickle_data(pickle_file):
  import pickle
  with open(pickle_file, 'rb') as f:
      data = pickle.load(f)
  return data