def square(preface: bool = True, n = 5):
    if preface:
        return "Hello Dr James" + str(n**2)
    else:
        return n**2

print(square(n=10))
    
