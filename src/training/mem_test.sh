data_dir="./mprof/data"
img_dir="./mprof/result"
file_name="2902_new_4"

if [ -d ${data_dir} ]; then
    echo Logging to ${data_dir}
else
    echo Making dir 
    mkdir ${data_dir}
fi 

if [ -d ${img_dir} ]; then
    echo Logging to ${img_dir}
else
    echo Making dir 
    mkdir ${img_dir}
fi
echo ${file_name}
mprof run -M -o ${data_dir}/${file_name} train_single_env.py 
mprof plot ${data_dir}/${file_name} -s -o ${img_dir}/${file_name} 
