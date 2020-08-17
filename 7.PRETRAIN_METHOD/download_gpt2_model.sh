FILE=./gpt_ckpt.zip
if test -f "$FILE"; then
    echo "$FILE not exists."
    wget https://www.dropbox.com/s/nzfa9xpzm4edp6o/gpt_ckpt.zip
else
    echo "$FILE exists."
fi
unzip "$FILE"
rm "$FILE"
