ZIP_FILE=./gpt_ckpt.zip
DIR_PATH=./gpt_ckpt
if test -f "$DIR_PATH"; then
    if test -f "$ZIP_FILE"; then
        echo "$ZIP_FILE not exists."
        wget https://www.dropbox.com/s/nzfa9xpzm4edp6o/gpt_ckpt.zip
    else
        echo "$ZIP_FILE exists."
    fi
    unzip "$FILE"
    rm "$FILE"
else
    echo "$DIR_PATH exists."
fi
