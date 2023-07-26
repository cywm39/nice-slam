import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue
import time
import numpy as np

def producer(queue, mark):
    # for item in data:
    #     queue.put(item)
    # queue.put(None)  # Use None as the sentinel to signal the end of data
    tmp = torch.tensor([1.1,2.2,3.3,4.4,5.5])
    queue.put({'type': 'coarse', 'grid': tmp.numpy()})
    # queue.put(None)
    mark[0] = True
    time.sleep(10)
    tmp[0] = 6.6

def consumer(queue, mark):
    # while True:
    #     item = queue.get()
    #     if item is None:
    #         break  # Break the loop when encountering the sentinel
    #     print("Consumer got:", item)
    while mark[0] == False:
        continue
    # while True:
    #     item = queue.get()
    #     if item is None:
    #         break
    #     print("consumer got: ", item)
    tmp = queue.get()
    print("consumer first got: ", tmp)
    mark[0] = False
    time.sleep(15)
    print("tmp now: ", tmp)

if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    # mark = torch.tensor([False])
    # mark.share_memory_()
    # # Create a shared queue
    # queue = Queue()
    

    # # Data to be sent to the consumer
    # # data = [1, 2, 3, 4, 5]

    # # Create two processes, one for producer and the other for consumer
    # p_producer = Process(target=producer, args=(queue, mark))
    # p_consumer = Process(target=consumer, args=(queue, mark))

    # # Start both processes
    # p_producer.start()
    # p_consumer.start()

    # # Wait for both processes to finish
    # p_producer.join()
    # p_consumer.join()
    # print("main: ", mark[0])
    x = torch.zeros([2,3])
    print(x)
    y = torch.ones([2,2])
    print(y)
    result = torch.cat([x, y], dim=-1)
    print(result)