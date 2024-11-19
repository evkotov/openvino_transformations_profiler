import sys
from typing import List


def fix_lines(lines: List[str]) -> List[str]:
    header = lines[0].split(';')
    iteration_idx = header.index('iteration')
    type_idx = header.index('type')
    new_lines = [lines[0]]
    n_iter = 1
    prev_ir_path = None
    for line in lines[1:]:
        fields = line.split(';')
        ir_path = fields[1]
        if prev_ir_path is None or ir_path != prev_ir_path:
            prev_ir_path = ir_path
            n_iter = 1
            fields[iteration_idx] = str(n_iter)
        elif fields[type_idx] == 'compile_time':
            fields[iteration_idx] = str(n_iter)
            n_iter += 1
        else:
            fields[iteration_idx] = str(n_iter)
        new_lines.append(';'.join(fields))
    return new_lines


def fix_bug_pr2756(input_files: List[str]):
    for input_file in input_files:
        lines = None
        with open(input_file) as f:
            lines = f.readlines()
        lines = fix_lines(lines)
        output_file = input_file + '.fixed'
        with open(output_file, 'w') as f:
            for line in lines:
                f.write(line)


if __name__ == '__main__':
    fix_bug_pr2756(sys.argv[1:])
