from collections import Counter
import re
def alpha_filter(w):
    '''
    pattern to match a word of alphabetical characters.
    Can be adjusted to be tolerant to several other characters like ! and digits
    '''
    pattern = re.compile('^[!_A-Za-z]+$') # the ! character is meaningful
    if pattern.match(w):
        return True
    else:
        return False

def gen_vocabulary(lst, cnt):
	'''
	input a list of tokens, cnt is the number words we need 
	'''
	c = Counter(lst)
	dist = c.most_common(cnt) # distribution
	return [e[0] for e in dist]