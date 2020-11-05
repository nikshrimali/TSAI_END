# Advanced Python for General Computing & NLP

## Problem Statement

- Write a function using only list filter lambda that can tell whether a number is a Fibonacci number or not. You can use a pre-calculated list/dict to store fab numbers till 10000 PTS:100
- Using list comprehension (and zip/lambda/etc if required) write an expression that: PTS:100
- add 2 iterables a and b such that a is even and b is odd
- strips every vowel from a string provided (tsai>>t s)
- acts like a ReLU function for a 1D array
- acts like a sigmoid function for a 1D array
- takes a small character string and shifts all characters by 5 (handle boundary conditions) tsai>>yxfn
- A list comprehension expression that takes a ~200 word paragraph, and checks whether it has any of the swear words mentioned in https://github.com/RobertJGabriel/Google-profanity-words/blob/master/list.txt PTS:200 (Links to an external site.)
Using reduce function: PTS:100
- add only even numbers in a list
- find the biggest character in a string (printable ascii characters)
- adds every 3rd number in a list
- Using randint, random.choice and list comprehensions, write an expression that generates 15 random KADDAADDDD number plates, where KA is fixed, D stands for a digit, and A stands for Capital alphabets. 10<<DD<<99 & 1000<<DDDD<<9999 PTS:100
- Write the above again from scratch where KA can be changed to DL, and 1000/9999 ranges can be provided. Now use a partial function such that 1000/9999 are hardcoded, but KA can be provided PTS:100
 

#  Functions Implemented 

## def add_even_odd_list(l1:list,l2:list)-> list:
    '''adds 2 iterables a and b such that a is even and b is odd'''

## def strip_vowels(input_str:str)->str:
    '''strips every vowel from a string provided'''

## def check_fibb(t):
    '''A function using only list filter lambda that can tell whether a number is a Fibonacci number or not Using a pre-calculated list to store fab numbers till 10000'''

## def relu_list(input_list:list)->list:
    '''acts like a ReLU function for a 1D array'''

## def getbiggestchar(test:str)->str:
    '''Outputs the biggest char in the string provided'''

## def relu_list(input_list:list)->list:
    '''acts like a ReLU function for a 1D array'''

## def sumevenchar(input_list):
    '''adds only even numbers in a list'''

## def get_numplate()-> list:
    '''Using randint, random.choice and list comprehensions, write an expression that generates 15 random KADDAADDDD number plates, where KA is fixed, D stands for a digit, and A stands for Capital alphabets. 10<<DD<<99 & 1000<<DDDD<<9999'''

## def partial_numplate(st_code, range_start:int, range_end:int)-> list:
    '''Write the above again from scratch where KA can be changed to DL, and 1000/9999 ranges can be provided. Now use a partial function such that 1000/9999 are hardcoded, but KA can be provided'''

## def check_profane_words(input_para:str)->bool:
    '''A list comprehension expression that takes a ~200 word paragraph, and checks whether it has any of the swear words mentioned in '''
