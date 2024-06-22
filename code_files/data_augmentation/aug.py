import re



def replace_for_with_while(c_code):
    loop_counter = 0
    dic = {} # Dictionary to store the loop_counter and the number of closing braces to wait for adding the increment statement
    lines = c_code.split('\n')
    new_code = []
    for line in lines:
        if 'for (' in line:
            indent = ' ' * (len(line) - len(line.lstrip()))
            # Using regular expressions to capture the components of the for loop
            match = re.search(r'for\s*\(([^;]*);([^;]*);([^)]*)\)', line)
            if match:
                init = match.group(1).strip()
                cond = match.group(2).strip()
                inc = match.group(3).strip()
                if not init.startswith('int'):
                    init = f'int {init}'
                if inc[0] == '+':
                    number = int(init.split('=')[1].strip()) + 1
                    init = re.sub(r'\d+$', '', init)
                    init += f'{number}'
                new_code.append(indent + init + ";")  # Initialization statement
                new_code.append(indent + f'while ({cond}) {{')  # While loop start
                for k, v in dic.items():
                    v[1] += 1
                dic[loop_counter] = [inc, 1]
                loop_counter += 1
                continue
            else:
                new_code.append(line)  # For loops that don't match the simple pattern
        elif '}' in line:
            num_of_braces = line.count('}')
            key_to_delete = -1
            for k, v in dic.items():
                v[1] -= num_of_braces
                if v[1] == 0:
                    indent = ' ' * (len(line) - len(line.lstrip()))
                    line, end = line.split("}", 1)
                    new_code.append(line)
                    new_code.append(indent + v[0] + ";" + "}" + end)
                    key_to_delete = k
            if key_to_delete != -1:
                del dic[key_to_delete]
            else:
                new_code.append(line)
        else:
            new_code.append(line)
            if "{" in line:
                for k, v in dic.items():
                    v[1] += 1
    return '\n'.join(new_code)





# def replace_while_with_for(c_code):
#     lines = c_code.split('\n')
#     new_code = []
#     while_counter = 0
#     dic = {} # Dictionary to store the loop_counter and the number of closing braces to wait for adding the increment statement
#     for i, line in enumerate(lines):
#         stripped_line = line.strip()
#         if 'while (' in stripped_line:
#             while_counter += 1
#             cond = re.search(r'while\s*\((.*)\)', stripped_line).group(1).strip()
#             # Assume a typical integer initialization if not specified
#             init_var = re.search(r'(\w+)\s*<', cond)
#             init = f"int {init_var.group(1)} = 0;" if init_var else ""
#             # Detect the increment statement
#             increment = ""
#             for j in range(i + 1, len(lines)):
#                 next_line = lines[j].strip()
#                 increment = cond[0] + "++;"
#             # Replace while with for and adjust the initialization and increment
#             new_line = line.replace('while', 'for', 1).replace(cond, f"{init} {cond}; {increment}")
#             new_code.append(new_line)
            
#         elif '}' in line and len(dic) > 0:
#             # Check for the loop closure
#             new_code.append(line)
#         else:
#             new_code.append(line)

#     return '\n'.join([line for line in new_code if line.strip() != ''])



def count_for_loops(c_function_str):
    # This regular expression looks for the keyword 'for' surrounded by non-alphanumeric characters
    # It ignores 'for' in comments and strings by ensuring it is not inside quotes or after '//'
    pattern = r"(?<!['\"/.\w])for(?![\"'\w])"
    
    # Using re.findall to capture all occurrences that match the pattern
    # We then count the number of matches found
    for_loops = re.findall(pattern, c_function_str)
    
    return for_loops


def get_for_loop_lines(c_function_str):
    # Split the input string into lines
    lines = c_function_str.split('\n')
    
    # This pattern checks for the 'for' keyword as part of a loop declaration
    pattern = r"(?<!['\"/\w])for(?![\"'\w])"
    
    # Initialize an empty list to store lines containing 'for' loops
    for_loop_lines = []
    
    # Iterate through each line
    for line in lines:
        # Use re.search to find 'for' in the line
        if re.search(pattern, line):
            # Strip leading/trailing whitespace and add to the list
            for_loop_lines.append(line.strip())
    
    # Join all collected lines with a newline character and return
    return '\n'.join(for_loop_lines)


def problem_for_loop(c_function_str):
    lines = c_function_str.split('\n')  # Split the function into lines
    found_for_loop = False  # Flag to track if any 'for' loop is found
    count = 0
    for line in lines:
        line = line.strip()  # Strip whitespace from the current line
        if "for (" in line or "for(" in line:  # Check if 'for (' is in the line
            count += 1
            found_for_loop = True  # Mark that we found a 'for' loop
            # Attempt to extract the increment part of the for loop
            start = line.find("for (") + len("for (")
            end = line.find(")", start)
            if end == -1:  # If no closing parenthesis is found, continue to next line
                continue
            # Split the loop header into parts
            parts = line[start:end].split(';')
            if len(parts) < 3:  # If there are not enough parts, continue to next line
                continue
            increment = parts[2].strip()  # Get the increment part and strip spaces
            # Check if the increment part is an empty string or lacks typical increment operations
            if increment == '' or not any(op in increment for op in ("++", "--", "+=", "-=", "*=", "/=", "%=", "->", "+", "-", ">>", "<<")):
                return True, 0  # Return True if increment is not typical
    return False, count  # Return False if no such for loops are found


def count_while_loops(c_function_str):
    # This regular expression looks for the keyword 'while' surrounded by non-alphanumeric characters
    # It also ensures that 'while' isn't part of a larger word or inside quotes or comments
    pattern = r"(?<!['\"/.\w])while(?![\"'\w])"
    
    # Using re.findall to find all matches for the pattern
    while_loops = re.findall(pattern, c_function_str)
    
    return len(while_loops)


# Example C code with a for loop
c_code = """
void example() {
    for (int i = 0; i < 10; ++i) {
        printf("%d\\n", i);
        if (3 < 5) {
            continue;
        }
        while (t < 5) {
            t++;
        }
    }
    int y = 0;
    while (x < 5) {
        for (int z = 0; z < 10; ++z) {
            y++;
        }
        if (y == 5) {
            break;
        }
        else {
            continue;   
        x++;
    }
}
"""

# c_code = """
# void example() {
#     int y = 0;
#     while (x < 5) {
#         for (int z = 0; z < 10; z++) {
#             y++;
#         }
#         if (y == 5) {
#             break;
#         }
#         else {
#             continue;   
#         }
#         x++;
#     }
# }
# """

# print(replace_for_with_while(c_code))


# Number of function in trainset that has for loop - 1973
# Number of for loop in trainset - 3984
# Number of function with not good for loops that i can not do augmentations - 137

# Number of function in trainset that has while loop - 1610
# Number of while loop in trainset - 2809
# Number of function with while loop and without for loop - 871