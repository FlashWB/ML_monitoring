#!/bin/bash
dict_length=(500 1000 1500 2000 2500 3000)
models=(SVM_SVC_rbf GBDT SVM_SVC_linear)
#choose GaussianNB,MultinomialNB,BernoulliNB or SVM_SVC_linear SVM_SVC_rbf
vertorize=(onehot wf tf tfidf word2vec)
test_data_num=90000
train_data_num=10000
for model in ${models[@]}
do
    for vertor in ${vertorize[@]}
    do
	for length in ${dict_length[@]}
	do
	    cmd="python3 ./class.py $vertor $model ${length} 90000 10000 >> ./result.log"
	    echo $cmd
	    $cmd >> ./result.log
	done
    done
done


# ${dict_length[index]} 读取数组元素值
#`expr $a + $b` 两个变量运算加`exper `
# 关系运算符 -ef......
# '$name' 用单引号，原样输出字符串，输出：$name
# 输出结果定向至文件 >
# if ...;then ...; elif....; then ...;else ...;fi;
