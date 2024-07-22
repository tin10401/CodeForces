# Makefile

# Variables
TEMPLATE = template.cpp
SUBMIT = submit.cpp
INPUT = input.txt
OUTPUT = a.out

# Targets
.PHONY: new compile run

# Create new problem setup
new:
	cp $(TEMPLATE) $(SUBMIT)
	> $(INPUT)

# Compile the submit.cpp file
compile:
	g++ -o $(OUTPUT) $(SUBMIT)

# Run the compiled program with input from input.txt
run: compile
	./$(OUTPUT) < $(INPUT)

# Clean the output file
clean:
	rm -f $(OUTPUT)

