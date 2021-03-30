import sys
import hashlib

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, 'r') as inp:
        with open(output_file, 'w') as out:
            for line in inp:
                author_id, title, text = line.strip().split(';;;')
                author_id = hashlib.sha256(author_id.encode('utf-8')).hexdigest()
                print(author_id + ';;;' + title + ';;;' + text, file=out)

if __name__ == '__main__':
    main()
