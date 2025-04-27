# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g

# Executable name
TARGET = connected-four

# Source files
SRCS = main.c neural_network.c game.c

# Object files (automatically generated from SRCS)
OBJS = $(SRCS:.c=.o)

# Default target (build everything)
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile each .c file into .o file
%.o: %.c dlist.h
	$(CC) $(CFLAGS) -c $<

# Clean up build files
clean:
	rm -f $(OBJS) $(TARGET)

run:
	make && ./$(TARGET)

run-clean:
	run clean
# Mark targets that don't produce files
.PHONY: all clean
