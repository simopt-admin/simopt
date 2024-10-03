import os
import subprocess

file_names = []

cwd = os.getcwd()
test_folder = os.path.join(cwd, 'test')
for entry in os.listdir(test_folder):
    if entry.startswith('test_') and entry.endswith('.py'):
        file_names.append(entry)

runtime_dict = {}
for file_name in file_names:
    filepath = os.path.join(test_folder, file_name)
    class_name = file_name.split(".")[0]
    for method_name in ["test_run", "test_post_replicate", "test_post_normalize"]:
        test_name = f"{class_name}::{method_name}"
        print(f"Running test: {test_name}")
        command = f"pytest {filepath}::{test_name}"
        process = subprocess.run(command, shell=True, capture_output=True)
        # Look for the line with "1 passed in "
        output_lines = process.stdout.decode().split("\n")
        for line in output_lines:
            if "1 passed in" in line:
                runtime = line.split("1 passed in")[1]
                # Strip and remove anything that isn't a number or period
                runtime = "".join([c for c in runtime if c.isdigit() or c == "."])
                runtime_dict[test_name] = runtime
                print(f"Finished test in {runtime} seconds")

for key, value in runtime_dict.items():
    print(key, value)