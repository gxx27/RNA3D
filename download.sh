#!/bin/bash

urls=()

while IFS= read -r line; do
  urls+=("$line")
done < urls.txt

for url in "${urls[@]}"; do
  filename=$(basename "$url")

  wget "$url" -O "$filename"

  echo "finished download: $filename"

  if [[ $filename == *.zip ]]; then
    unzip "$filename"

    rm "$filename"

    echo "finished unzip: $filename"
  fi
done