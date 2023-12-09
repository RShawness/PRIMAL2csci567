#!/bin/bash

# Define a handler for SIGINT
handle_sigint() {
    echo "SIGINT received, exiting..."
    exit 0
}

# Set the trap for SIGINT
trap handle_sigint SIGINT

while true; do
    # Run your Python script
    python driver.py

    # Check exit status
    if [ $? -ne 0 ]; then
        echo "Script exited with error, restarting..."
        sleep 20 # Optional: to prevent immediate respawning
    else
        echo "Script completed successfully, exiting loop."
        break  # Exit the loop if the script exits normally
    fi
done
