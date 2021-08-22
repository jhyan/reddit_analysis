""" a = 3
def Fuc():
    global a
    print (a)  # 1
    a = a + 1
def b(a):
    print(a+1)
if __name__ == "__main__":
    print (a)  # 2
    a = a + 1
    Fuc()
    b(a) """
    
""" def main():
    print "%s" % foo

if __name__ == "__main__":
    foo = "bar"
    main() """
a = [1,2,3]
b = [(a,"jimmy","op")]
print(b)