import argparse
import numpy as np


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Testing a matcher")
    argparser.add_argument("input", type=str, help="input file")
    argparser.add_argument("output", type=str, help="output file")
    args = argparser.parse_args()

    results = np.load(args.input)
    f = open(args.output, 'w')
    for i in range(0,len(results)):
        result = results[i]
        for rank in range(0, len(result)):
            caption_id = result[rank]
            f.write(str(rank+1) + ' ' + str(i+1) + ' ' + str(caption_id+1) + '\n')

    f.close()

