# Makefile

# Variables
TEMPLATE = template.cpp
SUBMIT = submit.cpp
INPUT = input.txt
OUTPUT = a.out

# Targets
.PHONY: new run

# Create new problem setup
new:
	cp $(TEMPLATE) $(SUBMIT)
	> $(INPUT)

# Run the compiled program with input from input.txt and clean up afterwards
run:
	g++ -std=c++17 -DLOCAL -mcmodel=large -o $(OUTPUT) $(SUBMIT)
	./$(OUTPUT) < $(INPUT)
	rm -f $(OUTPUT)

