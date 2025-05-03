# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -g

# Directories
BUILD_DIR := target

# Executable name
TARGET = $(BUILD_DIR)/4nn

# Source files
SRCS = main.c ann.c game.c

# Object files in target directory
OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)

# Default target
all: $(TARGET)

# Link object files into executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile each .c file into .o in target directory
$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Ensure build directory exists
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean up
clean:
	rm -rf $(BUILD_DIR)

run:
	make && ./$(TARGET)

run-clean:
	make clean && make run

.PHONY: all clean run run-clean

