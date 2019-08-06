from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf



class customModelCheckpoint(Callback):
    def __init__(self, log_dir='../logs/', feed_inputs_display=None):
          super(customModelCheckpoint, self).__init__()
          self.seen = 0
          self.feed_inputs_display = feed_inputs_display
          self.writer = tf.summary.FileWriter(log_dir)


    def custom_set_feed_input_to_display(self, feed_inputs_display):
          self.feed_inputs_display = feed_inputs_display


    # A callback has access to its associated model through the class property self.model.
    def on_batch_end(self, batch, logs = None):
          logs = logs or {}
          self.seen += 1
          if self.seen % 8 == 0: # every 200 iterations or batches, plot the costumed images using TensorBorad;
              summary_str = []
              feature = self.feed_inputs_display[0][0]
              disp_gt = self.feed_inputs_display[0][1]
              disp_pred = self.model.predict_on_batch(feature)

              summary_str.append(tf.summary.image('disp_input/{}'.format(self.seen), feature, max_outputs=4))
              summary_str.append(tf.summary.image('disp_gt/{}'.format(self.seen), disp_gt, max_outputs=4))
              summary_str.append(tf.summary.image('disp_pred/{}'.format(self.seen), disp_pred, max_outputs=4))

              summary_st = tf.summary.merge(summary_str)
              summary_s = K.get_session().run(summary_st)
              self.writer.add_summary(summary_s, global_step=self.seen)
              self.writer.flush()

