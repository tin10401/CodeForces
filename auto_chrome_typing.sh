#!/bin/bash

x=1  # Starting value of x

# Focus Chrome window
chrome_window=$(xdotool search --onlyvisible --class chrome | head -1)
xdotool windowactivate "$chrome_window"

# Loop for automating the sequence
while true; do
  # Press Enter
  xdotool key Return
 
  # Press Ctrl+P (or any other control sequence)
  xdotool key ctrl+p
 
  # Press Enter again
  xdotool key Return
 
  # Type the current value of x
  xdotool type "$x"
 
  # Press Enter after typing x
  xdotool key Return
 
  # Increase x by 1 for the next iteration
  x=$((x + 1))
 
  # Sleep for a short time between iterations (adjust as needed)
  sleep 1
done
