# Variables
TEMPLATE = template.cpp
SUBMIT = submit.cpp
INPUT = input.txt
OUTPUT = a.out
FLAGS = -std=c++23 -DLOCAL

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

run2: $(SUBMIT)
	g++ $(FLAGS) -o $(OUTPUT) $(SUBMIT)
	rm output.txt
	./$(OUTPUT) < $(INPUT) >> output.txt
	rm -f $(OUTPUT)
	
# Compile only
compile: $(SUBMIT)
	g++ $(FLAGS) -o $(OUTPUT) $(SUBMIT)

# Clean up compiled files
clean:
	rm -f $(OUTPUT)

