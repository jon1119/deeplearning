import scipy.io
import numpy as np

# # 生成一个随机密钥 转换为uint8数组
key1 = np.frombuffer(np.random.bytes(256), dtype=np.uint8)
key2 = np.frombuffer(np.random.bytes(256), dtype=np.uint8)
key3 = np.frombuffer(np.random.bytes(256), dtype=np.uint8)
# # save file
mat_data1 = {'key': key1}
mat_data2 = {'key': key2}
mat_data3 = {'key': key3}


scipy.io.savemat('key.mat', mat_data1)
scipy.io.savemat('keyCb.mat', mat_data2)
scipy.io.savemat('keyCr.mat', mat_data3)

data = scipy.io.loadmat('..\keyCb.mat',struct_as_record=False, squeeze_me=True)
print(data.keys())

