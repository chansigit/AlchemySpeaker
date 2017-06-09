# ==============================================================================
# 构建训练集/测试集 图片编号

train_imgs = [x for x in range(1,1+8000)]
test_imgs = [x for x in range(8001,8001+1000)]
print("训练集: %d ~ %d" %(min(train_imgs), max(train_imgs)))
print("测试集: %d ~ %d" %(min(test_imgs),  max(test_imgs)) )

# ==============================================================================
# 处理caption，训练测试放一起
# 总词表
vocabSet = set()

filterSet=set(" ,.:""''`~!@#$%^&*()。，-+、＇：∶；?‘’“”〝〞ˆˇ﹕︰﹔﹖﹑·¨….¸;！´？！～—ˉ｜‖＂〃｀@﹫¡¿﹏﹋﹌︴々﹟#﹩$﹠&﹪%*﹡﹢﹦﹤‐￣¯―﹨ˆ˜﹍﹎+=<­­＿_-\ˇ~﹉﹊（）〈〉‹›﹛﹜『』〖〗［］《》〔〕{}「」【】︵︷︿︹︽_﹁﹃︻︶︸﹀︺︾ˉ﹂﹄︼")
def getNeatChars(dirtyChars):
    #tokens=jieba.cut(dirtySentns)
    tokens = list(dirtyChars)
    neatTokens = ""
    for token in tokens:
        # 检测该词是否和过滤集有交集
        if token in filterSet :# 如果有，这词是标点符号，扔掉
            continue
        else: # 非标点符号
            neatTokens += " "+token
            vocabSet.add(token)
    return neatTokens.strip()

# captionTrain是一个字典， key是图片编号，value是一个list，包含了一个图对应的captions
captionTrainFile = '/data/PRMilestoneProj/captions/train.txt'
imgID = -1 # Not begin yet
captionTrain=dict()
for line in open(captionTrainFile, 'r'):
    line=line.strip('\ufeff').strip()
    if line.isdigit(): # Meet a new image
        imgID=int(line)
        continue
    neatSentence = getNeatChars(line)   
    try:
        captionTrain[imgID] = captionTrain[imgID] + [neatSentence]
    except:
        captionTrain[imgID] = [neatSentence]

print("训练集+测试集总汉字数:",len(vocabSet))


# captionTest是一个字典， key是图片编号，value是一个list，包含了一个图对应的captions
captionTestFile = '/data/PRMilestoneProj/captions/valid.txt'        
imgID = -1 # Not begin yet
captionTest=dict()
for line in open(captionTestFile, 'r'):
    line=line.strip('\ufeff').strip()
    if line.isdigit(): # Meet a new image
        imgID=int(line)
        continue
    neatSentence = getNeatChars(line)   
    try:
        captionTest[imgID] = captionTest[imgID] + [neatSentence]
    except:
        captionTest[imgID] = [neatSentence]

# 把captionTrain, captionTest并在一起，就是data，
# 这里面包含了 训练集、测试集 所有的caption
data=captionTrain.copy()
data.update(captionTest)
print(data[1])
import pickle
with open( "captionData.pkl", "wb" ) as captionData_f:
    pickle.dump( data, captionData_f ) 


# ==============================================================================
#处理图像特征，训练/测试 分开  
import h5py
FC1FileName = "/data/PRMilestoneProj/CNNFeatures/image_vgg19_fc1_feature_78660340.h5"
FC2FileName = "/data/PRMilestoneProj/CNNFeatures/image_vgg19_fc2_feature_78660340.h5"

FC1 = h5py.File(FC1FileName, 'r')
FC2 = h5py.File(FC1FileName, 'r')
print("FC1, FC2的h5里包含以下数据集：")
print([x for x in FC1.keys()])
print([x for x in FC2.keys()])

FC1_train = FC1['train_set']
FC1_valid = FC1['validation_set']
FC1_test  = FC1['test_set']

FC2_train = FC2['train_set']
FC2_valid = FC2['validation_set']
FC2_test  = FC2['test_set']


encoded_images = {}
for img in train_imgs:
    encoded_images[img] = FC2_train[img-1]
for img in test_imgs:
    encoded_images[img] = FC2_test[img-8001]

import pickle
with open( "encoded_images.p", "wb" ) as pickle_f:
    pickle.dump( encoded_images, pickle_f ) 
    
print("图像FC2编码载入成功")
print("每个图像编码维度：",encoded_images[1].shape)
print("图像编码已存成'encoded_images.p'文件")

# ==============================================================================
#绑定图像特征和caption，构建训练集和测试集并写入文件
f_train_dataset_name = 'PR_train_dataset.txt'
f_test_dataset_name  = 'PR_test_dataset.txt'

f_train_dataset = open(f_train_dataset_name,'w')
f_train_dataset.write("image_id\tcaptions\n")

f_test_dataset = open(f_test_dataset_name,'w')
f_test_dataset.write("image_id\tcaptions\n")

c_train = 0 # 训练集数目
for img in train_imgs:
    for capt in data[img]:
        caption = "<start> "+capt+" <end>"
        f_train_dataset.write(str(img)+"\t"+caption+"\n")
        c_train += 1
f_train_dataset.close()


c_test = 0 # 测试集数目
for img in test_imgs:
    for capt in data[img]:
        caption = "<start> "+capt+" <end>"
        f_test_dataset.write(str(img)+"\t"+caption+"\n")
        c_test += 1
f_test_dataset.close()
print("训练集句子数 vs 测试集句子数 = %d vs %d"%(c_train, c_test))
print("中间文件'PR_train_dataset.txt', 'PR_test_dataset.txt' 已生成")

