import sys
import os

def main():
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    with open(input_file, 'r', encoding='utf-8') as inp:
        for line in inp:
            #Get data
            stripped = line.strip()
            author_id, title, text = stripped.split(';;;')
            output_file = os.path.join(output_folder, author_id) + '.txt'

            #Put redudant user names go into the different files
            '''if os.path.isfile(output_file + '.txt'):
                i = 1
                name = output_file + '.' + str(i) + '.txt'
                while os.path.isfile(name):
                    name = output_file + '.' + str(i) + '.txt'''

            #Put redudant user names go into the same file
            with open(output_file, 'a', encoding='utf-8') as out:
                print(stripped, file=out)

if __name__ == '__main__':
    main()
