import os
import json
import argparse

def read_file(args):
    for filename in os.listdir(args.path):
        f = os.path.join(args.path,filename)
        if os.path.isdir(f):
            for ffn in os.listdir(f):
                ff = os.path.join(args.path,ffn)

                if os.path.isfile(ff) and ff.endswith("json"):
                    with open(ff) as F:
                        result_json=json.load(F)
                        print("****", ff)
                        print(result_json)
                        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='jigsaw')
    parser.add_argument('--path', type=str, default='.')
    args = parser.parse_args()
    print(args)
    read_file(args)