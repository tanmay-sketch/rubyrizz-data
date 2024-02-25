
Copy code
#!/bin/bash

mkdir -p train

count=1
for file in *.HEIC; do
    heif-convert "$file" "train/image_${count}.jpg"
    rm "$file"  # Delete the original .HEIC file
    ((count++))
done