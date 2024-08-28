# Variables
TEMPLATE = template.cpp
SUBMIT = submit.cpp
INPUT = input.txt
OUTPUT = a.out
FLAGS = -std=c++20 -DLOCAL -mcmodel=large

# Targets
.PHONY: new run clean compile

# Create new problem setup
new:
	cp $(TEMPLATE) $(SUBMIT)
	> $(INPUT)

run: $(SUBMIT)
	g++ $(FLAGS) -o $(OUTPUT) $(SUBMIT)
	./$(OUTPUT) < $(INPUT)
	rm -f $(OUTPUT)

# Compile only
compile: $(SUBMIT)
	g++ $(FLAGS) -o $(OUTPUT) $(SUBMIT)

# Clean up compiled files
clean:
	rm -f $(OUTPUT)

