# Variables
TEMPLATE = template.cpp
SUBMIT = submit.cpp
GENERATOR = generator.cpp
INPUT = input.txt
OUTPUT = a.out
FLAGS = -std=c++23 -O2 -DLOCAL -I./ac-library

# Targets
.PHONY: new run clean run2

# Create new problem setup
new:
	@cp $(TEMPLATE) $(SUBMIT)
	@> $(INPUT)

# Run with input redirection
run: $(SUBMIT)
	@g++ $(FLAGS) -o $(OUTPUT) $(SUBMIT)
	@./$(OUTPUT) < $(INPUT)
	@rm -f $(OUTPUT)

# Run generator and then run the submit solution
run2: $(GENERATOR) $(SUBMIT)
	@g++ $(FLAGS) -o generator $(GENERATOR)
	@./generator > $(INPUT)
	@g++ $(FLAGS) -o $(OUTPUT) $(SUBMIT)
	@./$(OUTPUT) < $(INPUT)
	@rm -f $(OUTPUT) generator

# Clean up compiled files
clean:
	@rm -f $(OUTPUT) generator

