import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.Session as sess:
            ...

            # Saving
            inputs = {
                "batch_size_placeholder": batch_size_placeholder,
                "features_placeholder": features_placeholder,
                "labels_placeholder": labels_placeholder,
            }
            outputs = {"prediction": model_output}
            tf.saved_model.simple_save(
                sess, 'path/to/your/location/', inputs, outputs
            )


with tf.Graph().as_default() as graph:
    with tf.Session as sess:
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            'path/to/your/location/',
        )
        batch_size_placeholder = graph.get_tensor_by_name('batch_size_placeholder:0')
        features_placeholder = graph.get_tensor_by_name('features_placeholder:0')
        labels_placeholder = graph.get_tensor_by_name('labels_placeholder:0')
        prediction = restored_graph.get_tensor_by_name('dense/BiasAdd:0')

        sess.run(prediction, feed_dict={
            batch_size_placeholder: some_value,
            features_placeholder: some_other_value,
            labels_placeholder: another_value
        })
