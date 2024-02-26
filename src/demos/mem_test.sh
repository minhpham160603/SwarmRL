data_dir="./mprof/data"
img_dir="./mprof/result"
file_name="run_2_no_del_testing"

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

mprof run -o ${data_dir}/${file_name} demo_single_env.py 

mprof plot ${data_dir}/${file_name} -o ${img_dir}/${file_name} 