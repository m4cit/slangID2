import fileinput

del_symbols = "€´#$%&\'()*+-./:;<=>?@[\\]^_`{|}~"

# removes unwanted symbols and changes everything into lowercase
def pp(file):
    for symbol in del_symbols:
        for line in fileinput.input(file, inplace=1):
            print(line.replace(symbol, "").lower(), end='')
    print("\nData was processed!")
