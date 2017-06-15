# ==============================================================================
# 构建训练集/测试集 图片编号

train_imgs = [x for x in range(   1,   1+8000)]
test_imgs  = [x for x in range(8001,8001+1000)]
sub_imgs   = [x for x in range(9001,9001+1000)]
print("训练集: %d ~ %d" %(min(train_imgs), max(train_imgs)))
print("测试集: %d ~ %d" %(min(test_imgs),  max(test_imgs)) )
print("提交集: %d ~ %d" %(min(sub_imgs),   max(sub_imgs)) )

# ==============================================================================
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
print(data[2])

fp=open("Automation.token.txt","w")
ToT= len(data)
for i in range(ToT):
    #print(i+1,data[i+1])
    capCnt=len(data[i+1])
    for capID in range(capCnt):
        fp.write("%d#%d\t%s\n"%(i+1, capID, data[i+1][capID]) )
fp.close()
    
fp=open("Automation.trainImages.txt","w")
for i in range(1,1+8000):
    fp.write(str(i)+"\n")
fp.close()

fp=open("Automation.testImages.txt","w")
for i in range(8001,8001+1000):
    fp.write(str(i)+"\n")
fp.close()

fp=open("Automation.subImages.txt","w")
for i in range(9001,9001+1000):
    fp.write(str(i)+"\n")
fp.close()
# ==========================================================================
#处理图像特征，训练/测试 分开  
import h5py

FC2FileName = "/data/PRMilestoneProj/CNNFeatures/image_vgg19_fc2_feature_677004464.h5"
FC2 = h5py.File(FC2FileName, 'r')
print("FC2的h5里包含以下数据集：")

print([x for x in FC2.keys()])
FC2_train = FC2['train_set']
FC2_valid = FC2['validation_set']
FC2_test  = FC2['test_set']

encoded_images = {}
for img in train_imgs:
    encoded_images[str(img)] = FC2_train[img-1]
for img in test_imgs:
    encoded_images[str(img)] = FC2_valid[img-8001]
for img in sub_imgs:
    encoded_images[str(img)] = FC2_test[img-9001]

import pickle
with open( "temp/encoded_images.p", "wb" ) as pickle_f:
    pickle.dump( encoded_images, pickle_f ) 
    
print("图像FC2编码载入成功")
print("每个图像编码维度：",encoded_images['1'].shape)
print("图像编码已存成'encoded_images.p'文件")

