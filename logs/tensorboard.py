import threading
import os
def launch():
    os.system('tensorboard --logdir=logs'+' --port=8888')
    return

t = threading.Thread(target=launch, args=([]))
t.start()