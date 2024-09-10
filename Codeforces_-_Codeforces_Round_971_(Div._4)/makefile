# Compiler settings
CXX = g++
CXXFLAGS = -std=c++20 -O2 -Wall

# Default target
.PHONY: run clean

# Extract the problem name (without .cpp extension)
PROBLEM = $(basename $(lastword $(MAKECMDGOALS)))

# Run the selected problem with its corresponding input and compare outputs
run: $(PROBLEM)
	$(CXX) $(CXXFLAGS) -o $(PROBLEM) $(PROBLEM).cpp
	@echo "Running $(PROBLEM) with $(PROBLEM).txt..."
	./$(PROBLEM) < $(PROBLEM).txt > $(PROBLEM)_output.txt
	@if [ -f $(PROBLEM)output.txt ]; then \
		if diff -u $(PROBLEM)_output.txt $(PROBLEM)output.txt; then \
			echo "Test passed!"; \
		else \
			echo "Test failed!"; \
		fi; \
	else \
		echo "No expected output file $(PROBLEM)output.txt found."; \
	fi

# Clean the generated files
clean:
	rm -f $(PROBLEM) $(PROBLEM)_output.txt

# Prevent make from treating run and clean as files
%:
	@:

