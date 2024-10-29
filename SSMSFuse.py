import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as ly
from tensorflow.keras.layers import Conv2D, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Reshape
import tensorflow.keras.layers as layers

class SSJAM(tf.keras.Model):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(SSJAM, self).__init__()
        self.channel = channel
        self.conv0 = tf.keras.Sequential([
            Conv2D(self.channel, (1,1),
                          padding='valid'),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2)
        ])

        # Channel attention
        self.max_pool = GlobalMaxPooling2D()
        self.avg_pool = GlobalAveragePooling2D()

        self.mlp = tf.keras.Sequential([
            Dense(channel // reduction, use_bias=False),
            tf.keras.layers.Activation('relu'),
            Dense(channel, use_bias=False)
        ])

        # Spatial attention
        self.conv = Conv2D(1, (spatial_kernel, spatial_kernel), padding='same', use_bias=False)
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

    def call(self, S ,F):
        map = tf.concat([S, F], axis=3)
        map0 = self.conv0(map)
        # 计算通道注意力
        max_out = self.mlp(Reshape((1, 1, self.channel))(self.max_pool(F)))
        avg_out = self.mlp(Reshape((1, 1, self.channel))(self.avg_pool(F)))
        channel_out = self.sigmoid(max_out + avg_out)
        x = Multiply()([channel_out, map0])

        # 计算空间注意力
        max_out = tf.reduce_max(S, axis=3, keepdims=True)
        avg_out = tf.reduce_mean(S, axis=3, keepdims=True)
        spatial_out = self.sigmoid(self.conv(tf.concat([max_out, avg_out], axis=3)))
        x = Multiply()([spatial_out, x])
        return x

def upSampleZ(X):
    weight_decay = 1e-4
    upX_0 = tf.layers.conv2d_transpose(X, filters=110, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                       activation=tf.nn.leaky_relu,
                                       kernel_initializer=ly.variance_scaling_initializer(),
                                       kernel_regularizer=ly.l2_regularizer(weight_decay))

    upX_1 = tf.layers.conv2d_transpose(upX_0, filters=110, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                       activation=tf.nn.leaky_relu,
                                       kernel_initializer=ly.variance_scaling_initializer(),
                                       kernel_regularizer=ly.l2_regularizer(weight_decay))
    upX_2 = tf.layers.conv2d_transpose(upX_1, filters=110, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                       activation=tf.nn.leaky_relu,
                                       kernel_initializer=ly.variance_scaling_initializer(),
                                       kernel_regularizer=ly.l2_regularizer(weight_decay))
    upX_3 = tf.layers.conv2d_transpose(upX_2, filters=110, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                       activation=tf.nn.leaky_relu,
                                       kernel_initializer=ly.variance_scaling_initializer(),
                                       kernel_regularizer=ly.l2_regularizer(weight_decay))
    upX_4 = tf.layers.conv2d_transpose(upX_3, filters=110, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                       activation=tf.nn.leaky_relu,
                                       kernel_initializer=ly.variance_scaling_initializer(),
                                       kernel_regularizer=ly.l2_regularizer(weight_decay))

    return upX_4

def upSample(X,outdim):
    weight_decay = 1e-4
    upX_0 = tf.layers.conv2d_transpose(X, filters=outdim, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                       activation=tf.nn.leaky_relu,
                                       kernel_initializer=ly.variance_scaling_initializer(),
                                       kernel_regularizer=ly.l2_regularizer(weight_decay))


    return upX_0

def pos_emb(input,outdim):
    # print(input.shape)
    p1 = ly.conv2d(input,outdim ,kernel_size=3,stride=1,
          activation_fn=tf.nn.leaky_relu)
    out = ly.conv2d(p1,outdim,kernel_size=3,stride=1,
          activation_fn=None)
    return out

def spectralformer(input):

    dim_b = input.shape[0]
    dim_h = input.shape[1]
    dim_w = input.shape[2]
    dim_c = input.shape[3]
    dim_head = 64
    heads = 16
    input = tf.reshape(input, [dim_b,dim_h*dim_w,dim_c])

    fc = tf.keras.layers.Dense(dim_head*heads,  use_bias=False)  #(_,h*w,8*64)

    q_inp = fc(input)
    k_inp = fc(input)  # (_,h*w,8*64)
    v_inp = fc(input)
    # (_,8,h*w,64)
    q = tf.transpose(tf.reshape(q_inp,[dim_b,dim_h*dim_w,heads,dim_head]), perm=[0,2,1,3])  # (_,8,h*w,64)
    k = tf.transpose(tf.reshape(k_inp, [dim_b,dim_h * dim_w, heads, dim_head]), perm=[0,2,1,3])
    v = tf.transpose(tf.reshape(v_inp, [dim_b,dim_h * dim_w, heads, dim_head]), perm=[0,2,1,3])
    # (_,8,64,h*w)
    q = tf.transpose(q, perm=[0,1,3,2]) # (_,8,64,h*w)
    k = tf.transpose(k, perm=[0,1,3,2])
    v = tf.transpose(v, perm=[0,1,3,2])

    q = tf.linalg.l2_normalize(q, axis=-1)
    k = tf.linalg.l2_normalize(k, axis=-1)
    q = tf.transpose(q, perm=[0,1,3,2]) # (_,8,h*w,64)
    attn = k @ q  # [_,8,64,64]
    attn = tf.nn.softmax(attn, axis=-1)
    x = attn @ v  # [_,8,64,h*w]
    x = tf.transpose(x, perm=[0,1,3,2])  #[_,h*w,8,64]  #perm=[0,1,3,2]
    x = tf.reshape(x, [ dim_b,dim_h * dim_w, heads * dim_head])  # [_,h*w,8*64]
    fc_out = tf.keras.layers.Dense(dim_c, use_bias=False)  # (_,h*w,8*64)
    # print(x.shape, v_inp.shape)

    # fp_out = tf.transpose(tf.reshape(v_inp, [dim_b, dim_h,dim_w, heads*dim_head]), perm=[0, 3, 1, 2])
    fp_out = tf.reshape(v_inp, [dim_b, dim_h, dim_w, heads * dim_head])
    fp_out = pos_emb(fp_out, outdim=46)  #位置嵌入
    x = tf.reshape(fc_out(x), [dim_b,dim_h, dim_w, dim_c])
    x = x+fp_out

    return x

def spectralnet(x,dim):
    weight_decay = 1e-4
    x1 = ly.conv2d(x, dim , kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))  # 512
    x2 = ly.conv2d(x1, dim, kernel_size=3, stride=1,
                  activation_fn=tf.nn.leaky_relu,
                  weights_initializer=ly.variance_scaling_initializer(),
                  weights_regularizer=ly.l2_regularizer(weight_decay))  # 512
    x2 = x2+x1
    x3 = ly.conv2d(x2, dim, kernel_size=3, stride=1,
                   activation_fn=tf.nn.leaky_relu,
                   weights_initializer=ly.variance_scaling_initializer(),
                   weights_regularizer=ly.l2_regularizer(weight_decay))  # 512
    x3 = x3 + x1
    x3 = spectralformer(x3)
    x3 = upSample(x3,dim)
    return x3

def model(Y,inZ,outdim=46):
    weight_decay = 1e-4

    UPZ = upSampleZ(inZ)
    Y = tf.concat([UPZ,Y],axis=3)

    conv1_1 = ly.conv2d(Y, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
    conv1_2 = ly.conv2d(conv1_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    conv2_1 = ly.conv2d(conv1_2, outdim, kernel_size=3, stride=2,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
    conv2_2 = ly.conv2d(conv2_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    conv3_1 = ly.conv2d(conv2_2, outdim, kernel_size=3, stride=2,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
    conv3_2 = ly.conv2d(conv3_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    conv4_1 = ly.conv2d(conv3_2, outdim, kernel_size=3, stride=2,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
    conv4_2 = ly.conv2d(conv4_1,outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    conv5_1 = ly.conv2d(conv4_2, outdim, kernel_size=3, stride=2,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
    conv5_2 = ly.conv2d(conv5_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    conv6_1 = tf.layers.conv2d_transpose(conv5_2, outdim, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=ly.variance_scaling_initializer(),
                                         kernel_regularizer=ly.l2_regularizer(weight_decay))
    conv6_2 = ly.conv2d(conv6_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    Z = upSample(inZ,outdim)
    Z = spectralnet(Z,dim = outdim)

    conv6_2 = tf.concat([conv4_2,conv6_2],axis=3)

    conv6_2 = SSJAM(channel=outdim, reduction=16, spatial_kernel=7)(conv6_2, Z)

    conv7_1 = tf.layers.conv2d_transpose(conv6_2,outdim, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=ly.variance_scaling_initializer(),
                                         kernel_regularizer=ly.l2_regularizer(weight_decay))
    conv7_2 = ly.conv2d(conv7_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    Z = spectralnet(Z,dim = outdim)

    conv7_2 = tf.concat([conv3_2, conv7_2], axis=3)
    conv7_2 = SSJAM(channel=outdim, reduction=16, spatial_kernel=7)(conv7_2, Z)

    conv8_1 = tf.layers.conv2d_transpose(conv7_2, outdim, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=ly.variance_scaling_initializer(),
                                         kernel_regularizer=ly.l2_regularizer(weight_decay))
    conv8_2 = ly.conv2d(conv8_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    Z = spectralnet(Z,dim = outdim)

    conv8_2 = tf.concat([conv2_2, conv8_2], axis=3)
    conv8_2 = SSJAM(channel=outdim, reduction=16, spatial_kernel=7)(conv8_2, Z)

    conv9_1 = tf.layers.conv2d_transpose(conv8_2, outdim, kernel_size=(2, 2), strides=(2, 2), padding='valid',
                                         activation=tf.nn.leaky_relu,
                                         kernel_initializer=ly.variance_scaling_initializer(),
                                         kernel_regularizer=ly.l2_regularizer(weight_decay))
    conv9_2 = ly.conv2d(conv9_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    Z = spectralnet(Z,dim = outdim)

    conv9_2 = tf.concat([conv1_2, conv9_2], axis=3)
    conv9_2 = SSJAM(channel=outdim, reduction=16, spatial_kernel=7)(conv9_2, Z)

    conv10_1 = ly.conv2d(conv9_2 , outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))
    conv10_2 = ly.conv2d(conv10_1, outdim, kernel_size=3, stride=1,
                        activation_fn=tf.nn.leaky_relu,
                        weights_initializer=ly.variance_scaling_initializer(),
                        weights_regularizer=ly.l2_regularizer(weight_decay))

    return conv10_2