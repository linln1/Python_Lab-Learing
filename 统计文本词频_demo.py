# CalHamletV1.py


def getText():
    fr = open("hamlet.txt","r").read()
    fr = fr.lower()
    for ch in '!"#$%&()*+,_./:;<=>?@[\\]^_‘{|}~':
        fr = fr.replace(ch, " ")
    return fr

def main():
    hamletTxt =getText()
    words = hamletTxt.split()
    counts = {}
    for word in words:
        counts[word] = counts.get(word, 0)+1
    items = list(counts.items())
    items.sort(key = lambda x:x[1] ,reverse=True)
    for i in range(100):
        word ,count = items[i]
        print("{0:<10}{1:>5}".format(word, count))

main()
