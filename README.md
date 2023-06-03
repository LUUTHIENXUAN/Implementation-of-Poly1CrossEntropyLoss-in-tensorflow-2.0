# Implementation-of-Poly1CrossEntropyLoss-in-tensorflow-2.0

```
class Poly1CrossEntropyLoss(tf.keras.losses.Loss):
    
    def __init__(self, num_classes, epsilon=1.0, reduction="sum", weight=None, name="poly_cross_entropy", **kwargs):
        super(Poly1CrossEntropyLoss, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.weight = weight

    def call(self, labels, logits):
        
        labels_onehot = tf.one_hot(labels, depth=self.num_classes, dtype=logits.dtype) 
        pt = tf.reduce_sum(labels_onehot * tf.nn.softmax(logits, axis=-1), axis=-1)    
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits) #, from_logits=True
        poly1 = ce_loss + self.epsilon * (1 - pt)

        if self.reduction == "mean":
            poly1 = tf.reduce_mean(poly1)
        elif self.reduction == "sum":
            poly1 = tf.reduce_sum(poly1)

        return poly1

    def get_config(self):
        config = {
            'weight': self.weight,
            'epsilon': self.epsilon
        }
        base_config = super().get_config()
        return {**base_config, **config} 
```
