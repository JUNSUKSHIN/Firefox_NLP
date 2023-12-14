file_path = 'data.txt'

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

processed_lines = [line.strip().strip('"').strip(',') for line in lines]

with open(file_path, 'w', encoding='utf-8') as file:
    for line in processed_lines:
        file.write(line + '\n')
