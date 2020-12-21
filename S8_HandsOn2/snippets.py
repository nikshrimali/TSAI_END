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