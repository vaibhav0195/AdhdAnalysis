import sys
import hashlib

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    with open(input_file, 'r', encoding='utf-8') as inp:
        with open(output_file, 'w', encoding='utf-8') as out:
            for line in inp:
                segemnts = line.strip().split(';;;')
                if len(segemnts) != 3:
                    continue
                author_id, title, text = segemnts
                author_id = hashlib.sha256(author_id.encode('utf-8')).hexdigest()
                data = author_id + ';;;' + title + ';;;' + text
                print(data, file=out)

if __name__ == '__main__':
    main()
