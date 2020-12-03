import os
import subprocess
import json
# import redis
import numpy as np
import shutil


def execute(cmd, wait=True, printable=True):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if printable: print('[utils.execute] "%s"' % cmd)
    if wait:
        out, err = p.communicate()  # 等待程序运行，防止死锁
        out, err = out.decode('utf-8'), err.decode('utf-8')  # 从bytes转为str
        if err:
            raise ValueError(err)
        else:
            return out


def ensure_dirname(dirname, override=False):
    if os.path.exists(dirname) and override:
        print('[info] utils.ensure_dirname: removing dirname: %s' % os.path.abspath(dirname))
        shutil.rmtree(dirname)
    if not os.path.exists(dirname):
        print('[info] utils.ensure_dirname: making dirname: %s' % os.path.abspath(dirname))
        os.makedirs(dirname)

        # if override:
        #     shutil.rmtree(dirname)
        # if not os.path.exists(dirname):
        #     print('[info] utils.ensure_dirname: making dirname: %s' % os.path.abspath(dirname))
        #     os.makedirs(dirname)


# 使得json.dumps可以支持ndarray，直接cls=utils.JsonCustomEncoder
class JsonCustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# class Redis():
#     '''
#     Redis封装．使用方法：
#     r = Redis('rcnn')
#     r.set('0', np.array([1.2, 3.2]))
#     print(r.get('0'))
#
#     way表示方式，第一种json结构保留完好，速度慢
#     第二种str结构保留不好，需要指定shape，但速度非常快．
#     '''
#
#     def __init__(self, prefix, redis_dir='./redis', way='json', shape=None):
#         self.prefix = str(prefix)
#         self.redis_dir = os.path.abspath(redis_dir)
#         self.way = way
#         self.shape = shape
#         pid = execute('pgrep redis-server', printable=False)
#         if not pid:
#             ensure_dirname(self.redis_dir)
#             # config_filename = os.path.join(redis_dir, 'redis.conf')
#             stderr_filename = os.path.join(redis_dir, 'stderr.txt')
#             # execute('nohup redis-server %s 2> %s &' % (config_filename, stderr_filename), wait=False, printable=False)
#             execute('nohup redis-server 2> %s &' % (stderr_filename), wait=False, printable=False)
#             print('[utils.Redis]: Started redis-server, dirname is %s' % self.redis_dir)
#         else:
#             print('[utils.Redis]: redis-server already started, pid is %s.' % pid.strip())
#         self.db = redis.StrictRedis(host='127.0.0.1', port=6379, db=4)
#
#     def set(self, k, v, override=True):
#         key = self.prefix + str(k)
#         if self.way == 'json':
#             value = json.dumps(v, cls=JsonCustomEncoder)
#         elif self.way == 'str':
#             value = v.tostring()
#         else:
#             value = v
#         self.db.set(key, value, nx=not override)
#
#     def get(self, k, dtype=np.float32):
#         key = self.prefix + str(k)
#         value = self.db.get(key)
#         if self.way == 'json':
#             value = np.asarray(json.loads(value))
#
#         elif self.way == 'str':
#             value = np.frombuffer(value, dtype=dtype).reshape(self.shape)
#         return value
#
#
#     def __setitem__(self, key, value):
#         self.set(key, value)
#
#     def __getitem__(self, k):
#         return self.get(k)


if __name__ == "__main__":
    redis = Redis("feature", way="str", shape=(3,4))
    key = 188872
    value = np.random.random(size=(3,4))
    print(value.dtype)
    print(value.tostring())
    new_value = np.frombuffer(value.tostring(), dtype=np.float32)
    print(new_value)
    redis.set(key, value)
    print(redis.get(key))
