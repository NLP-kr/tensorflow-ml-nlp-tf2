FILE=./gpt_ckpt.zip
echo "$FILE download"

if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    wget https://www.dropbox.com/s/nzfa9xpzm4edp6o/gpt_ckpt.zip
fi

unzip "$FILE"
rm "$FILE"
