# Variables
TEMPLATE = template.cpp
SUBMIT = submit.cpp
INPUT = input.txt
OUTPUT = a.out
FLAGS = -std=c++23 -DLOCAL

# Targets
.PHONY: new run clean compile run2

# Create new problem setup
new:
	cp $(TEMPLATE) $(SUBMIT)
	> $(INPUT)

# Run with input redirection
run: $(SUBMIT)
	g++ $(FLAGS) -o $(OUTPUT) $(SUBMIT)
	./$(OUTPUT) < $(INPUT)
	rm -f $(OUTPUT)

# Run interactively for debugging
run2: $(SUBMIT)
	g++ $(FLAGS) -o $(OUTPUT) $(SUBMIT)
	./$(OUTPUT)

# Clean up compiled files
clean:
	rm -f $(OUTPUT)

