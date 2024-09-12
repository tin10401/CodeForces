import http.server
import socketserver
import json
import os
import shutil

PORT = 12345

# Counter for input and output files
input_counter = 0

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        global input_counter
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        # Define the contest directory based on the contest name
        contest_dir = data.get('group', 'default_contest').replace(' ', '_')
        if not os.path.exists(contest_dir):
            os.makedirs(contest_dir)

        # Copy and rename makefile_contest to makefile in the contest directory
        if os.path.exists('makefile_contest'):
            shutil.copy('makefile_contest', os.path.join(contest_dir, 'makefile'))
            print(f"Copied and renamed makefile_contest to makefile in {contest_dir}/")

        # Problem letter (A, B, C, etc.)
        problem_letter = chr(65 + input_counter)  # A=65 in ASCII
        problem_file = f"{problem_letter}.cpp"

        # Copy the template to the problem file
        if not os.path.exists(os.path.join(contest_dir, problem_file)):
            shutil.copy('template.cpp', os.path.join(contest_dir, problem_file))
            print(f"Created {problem_file} from template.cpp")

        # Save each input and output file with the problem-specific naming convention
        input_counter += 1
        input_file_path = os.path.join(contest_dir, f'{problem_letter}_input.txt')
        output_file_path = os.path.join(contest_dir, f'{problem_letter}_expected_output.txt')

        # Write the input data to a new file
        with open(input_file_path, 'w') as input_file:
            input_file.write(data['tests'][0]['input'])

        # Write the expected output to a new file
        with open(output_file_path, 'w') as output_file:
            output_file.write(data['tests'][0]['output'])

        print(f"Saved {problem_letter}_input.txt and {problem_letter}_expected_output.txt in {contest_dir}/")

        self.send_response(200)
        self.end_headers()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving on port {PORT}")
    httpd.serve_forever()

