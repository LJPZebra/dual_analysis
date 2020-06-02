import toml
import argparse

def check(pathList, concentrations, products, age):

  outList = []

  for path in pathList:
    subFile =""
    with open(path, 'r') as fsub:
      i = 0
      while i < 22:
          subFile += str(fsub.readline() + '\n')
          i += 1
    f = toml.loads(subFile)
    if products and not concentrations:
      if f["experiment"]["product"] in products and f["fish"]["age"] > int(age[0]) and f["fish"]["age"] < int(age[1]):
        outList.append(path)
    elif concentrations and not products:
      concentrations = [float(i) for i in concentrations]
      if f["experiment"]["concentration"] in concentrations and f["fish"]["age"] > int(age[0]) and f["fish"]["age"] < int(age[1]):
        outList.append(path)
    elif concentrations and products:
      concentrations = [float(i) for i in concentrations]
      if f["experiment"]["concentration"] in concentrations and f["experiment"]["product"] in products and f["fish"]["age"] > int(age[0]) and f["fish"]["age"] < int(age[1]):
        outList.append(path)
  
  out = ""
  for i in outList:
    out += i + " "

  return out





parser = argparse.ArgumentParser(description="Extract specific toml files from criterions")
parser.add_argument("path", nargs='+', help="List of toml files")
parser.add_argument("--concentration", dest="concentrations", nargs='+', help="Concentration criteria")
parser.add_argument("--product", dest="products",  nargs='+', help="Product criteria")
parser.add_argument("--age", dest="age", nargs='+', help="Fish age in days from to")

args = parser.parse_args()
a = check(args.path, args.concentrations, args.products, args.age)
print(a)
