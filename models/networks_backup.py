import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler


###############################################################################
# Helper Functions
###############################################################################

class Identity(nn.Module):
    def forward(self, x):
        return x


def get_scheduler(optimizer, opt):
    """
    Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    return scheduler


def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x):
            return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias=True):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


###################################################################################
## Your Implementation is from Here ##


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02,
             gpu_ids=[]):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'attention_basic':
        net = AttentionGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                 use_dropout=use_dropout)  # you can modify the input variables and types
    elif netG == 'basic':
        net = BaselineGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer,
                                use_dropout=use_dropout)  # you can modify the input variables and types
    elif netG == 'advanced':
        net = None  # you can modify the input variables and types (PA Step4)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, norm='instance', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'attention_basic':  # self attention based discriminator
        net = AttentionDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic':  # baseline discriminator
        net = BaselineDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'advanced':
        net = None  # you can modify the input variables and types (PA Step4)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image
        """
        super(GANLoss, self).__init__()
        ## Your Implementation Here ##
        self.gan_mode = gan_mode
        # fanchen
        # self.target_real_label = target_real_label
        # self.target_fake_label = target_fake_label
        # fanchen: need to reg as buffer
        self.register_buffer('target_real_label', torch.tensor(target_real_label))
        self.register_buffer('target_fake_label', torch.tensor(target_fake_label))

        if gan_mode == 'lsgan':
            # fanchen: lsgan uses L2 loss
            self.loss = nn.MSELoss()  ## Your Implementation Here ##
        elif gan_mode == 'vanilla':
            # fanchen: https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html
            # The loss function of the vanilla GAN measures the JS divergence between the distributions of pr and pg. This metric fails to provide a meaningful value when two distributions are disjoint.
            # Wasserstein metric is proposed to replace JS divergence because it has a much smoother value space. See more in the next section.
            self.loss = nn.BCEWithLogitsLoss()  ## Your Implementation Here ##
            # fanchen: binary_cross_entropy_with_logits
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        ## Your Implementation Here ##
        # fanchen: target vector making
        target = self.target_real_label if target_is_real else self.target_fake_label
        target = target.expand_as(prediction)
        return self.loss(prediction, target)


class BaselineGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False,
                 padding_type='reflect'):
        """Construct a generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            # fanchen: changing default norm layer to InstanceNorm2d, maybe unnecessary(?) as norm variable is given
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(BaselineGenerator, self).__init__()
        ## Your Implementation Here ##
        # fanchen: using "resnet_9blocks", according to the comments of define_G
        # In paper, it says
        # "Generator architectures We adopt our architectures
        # from Johnson et al. [23]. We use 6 residual blocks for
        # 128 × 128 training images, and 9 residual blocks for 256 ×
        # 256 or higher-resolution training images. Below, we follow
        # the naming convention used in the Johnson et al.’s Github
        # repository."
        # c7s1-64,d128,d256,R256,R256,R256,
        # R256,R256,R256,R256,R256,R256,u128
        # u64,c7s1-3
        # Let c7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1
        # dk denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2
        # Reflection padding was used to reduce artifacts
        # Rk denotes a residual block that contains two 3 × 3 convolutional layers with the same
        # number of filters on both layer
        # uk denotes a 3 × 3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters
        # and stride 1/2
        # A fractionally-strided convolution (deconvolution) a.k.a transposed convolution
        self.c7s1_64 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            norm_layer(ngf),
            nn.ReLU(True)  # fanchen: set inplace=True to save memory
        )
        self.d128 = nn.Sequential(
            # nn.ReflectionPad2d(3),  # fanchen: redundant
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        self.d256 = nn.Sequential(
            # nn.ReflectionPad2d(3), # fanchen: redundant
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )
        # self.R256_list = nn.ModuleList(
        #     [ResnetBlock(ngf * 4, padding_type, norm_layer, use_dropout) for _ in range(9)]
        # )
        self.R256_list = nn.Sequential(
            *[ResnetBlock(ngf * 4, padding_type, norm_layer, use_dropout) for _ in range(9)]
        )
        self.u128 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        self.u64 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )
        self.c7s1_3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            # norm_layer(ngf),
            # nn.ReLU(True)
            nn.Tanh()  # fanchen: using the implementation of the authors
        )

    def forward(self, input):
        """Standard forward"""
        ## Your Implementation Here ##
        x = self.c7s1_64(input)
        x = self.d128(x)
        x = self.d256(x)
        x = self.R256_list(x)
        x = self.u128(x)
        x = self.u64(x)
        x = self.c7s1_3(x)
        return x


class AttentionGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 padding_type='reflect'):
        """
        Construct generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """

        super(AttentionGenerator, self).__init__()
        ## Your Implementation Here ##
        # fanchen: baseline + attn
        self.attn_map_output = False
        self.c7s1_64 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            norm_layer(ngf),
            nn.ReLU(True)  # fanchen: set inplace=True to save memory
        )
        self.d128 = nn.Sequential(
            # nn.ReflectionPad2d(3),  # fanchen: redundant
            nn.Conv2d(ngf, ngf * 2, 3, stride=2, padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        # self.attn1 = Self_Attn(ngf * 2)  # fanchen: attn layer moved
        self.d256 = nn.Sequential(
            # nn.ReflectionPad2d(3), # fanchen: redundant
            nn.Conv2d(ngf * 2, ngf * 4, 3, stride=2, padding=1),
            norm_layer(ngf * 4),
            nn.ReLU(True)
        )
        self.attn1 = Self_Attn(ngf * 4)  # fanchen: attn layer moved
        # self.R256_list = nn.ModuleList(
        #     [ResnetBlock(ngf * 4, padding_type, norm_layer, use_dropout) for _ in range(9)]
        # )
        self.R256_list = nn.Sequential(
            *[ResnetBlock(ngf * 4, padding_type, norm_layer, use_dropout) for _ in range(9)]
        )
        self.u128 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf * 2),
            nn.ReLU(True)
        )
        self.u64 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 3, stride=2, padding=1, output_padding=1),
            norm_layer(ngf),
            nn.ReLU(True)
        )
        # self.attn2 = Self_Attn(ngf)
        self.c7s1_3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            # norm_layer(ngf),
            # nn.ReLU(True)
            nn.Tanh()  # fanchen: using the implementation of the authors
        )

    def forward(self, input):
        """Standard forward"""
        ## Your Implementation Here ##
        attn_map1 = attn_map2 = None
        x = self.c7s1_64(input)
        # print('after c7s1_64', x.size())
        x = self.d128(x)
        # print('after d128', x.size())
        # x, attn_map1 = self.attn1(x)
        x = self.d256(x)
        # x, attn_map1 = self.attn1(x)  # fanchen: attn removed for G
        # print('after d256', x.size())
        x = self.R256_list(x)
        # print('after R256', x.size())
        x = self.u128(x)
        # print('after u128', x.size())
        # fanchen: debug
        # for _ in range(100):
        #     print(x.size())
        # torch.Size([2, 128, 128, 128])
        # x, attn_map1 = self.attn1(x)
        x = self.u64(x)
        # print('after u64', x.size())
        # x, attn_map2 = self.attn2(x)
        x = self.c7s1_3(x)
        # print('final', x.size())

        if not self.attn_map_output:
            return x
        else:
            return x, attn_map1, attn_map2


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation=None):
        """
        in_dim   -- input feature's channel dim
        activation    -- activation function type
        """
        super(Self_Attn, self).__init__()
        ## Your Implementation Here ##
        # fanchen: input variables reg.
        self.in_dim = in_dim
        self.activation = activation
        # fanchen: https://arxiv.org/pdf/1805.08318.pdf
        # Figure 2, Eq. (1 - 4)
        # The image features from the previous hidden layer x ∈ R
        # C×N are first transformed into two feature spaces f, g
        # to calculate the attention, where f(x) = Wfx, g(x) = Wgx
        # In the above formulation, Wg ∈ R C¯×C ,Wf ∈ R C¯×C ,Wh ∈ R C¯×C , Wv ∈ R C×C¯
        # are the learned weight matrices, which are implemented as 1×1 convolutions.
        # For memory efficiency, we choose k = 8 (i.e., C¯ = C/8) in all our experiments.
        # In addition, we further multiply the output of the attention
        # layer by a scale parameter and add back the input feature
        # map. Therefore, the final output is given by, yi = γoi + xi
        # where γ is a learnable scalar and it is initialized as 0
        # @ Fig. 2, "The softmax operation is performed on each row"
        self.f = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.g = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.sm = nn.Softmax(-1)
        self.h = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.v = nn.Conv2d(in_dim // 8, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## Your Implementation Here ##
        # fanchen: recommended BCWH form
        B, C, W, H = x.size()
        N = W * H
        f_x = self.f(x).view(B, -1, N)  # fanchen: -1 supposed to be in_dim//8
        f_x_T = f_x.permute(0, 2, 1)
        g_x = self.g(x).view(B, -1, N)
        # fanchen: should use bmm instead of mm as batch-like inputs are used
        # fanchen: debug
        # print(f_x_T.size(), g_x.size())
        # torch.Size([2, 16384, 16])
        # torch.Size([2, 16, 16384])
        # torch.Size([2, 65536, 8])
        # torch.Size([2, 8, 65536])
        s = torch.bmm(f_x_T, g_x)  # fanchen: matrix S in the paper
        # fanchen: still ERROR!! @ s = torch.bmm(f_x_T, g_x), CUDA out of memory
        attn_map = self.sm(s).permute(0, 2, 1)  # fanchen: B * N * N, beta matrix in the paper
        # fanchen: ERROR!!
        # File "/home/CtrlDrive/fanchen/pyws/ee898_pa2/models/networks.py", line 509, in forward
        #    attn_map = self.sm(torch.bmm(f_x_T, g_x)).permute(0, 2, 1)  # fanchen: B * N * N, beta matrix in the paper
        # RuntimeError: CUDA out of memory.
        h_x = self.h(x).view(B, -1, N)
        # o = self.v(torch.bmm(h_x, attn_map)).view(B, C, W, H)  # fanchen: B * C * W * H
        o = self.v(torch.bmm(h_x, attn_map).view(B, -1, W, H))  # fanchen: B * C * W * H, modified @ 0617
        y = self.gamma * o + x
        if self.activation:  # fanchen: final activation
            y = self.activation(y)
        return y, attn_map


class BaselineDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(BaselineDiscriminator, self).__init__()
        ## Your Implementation Here ##
        # fanchen: using PatchGAN discriminator
        # For discriminator networks, we use 70 × 70 PatchGAN [22].
        # Let Ck denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer with k
        # filters and stride 2. After the last layer, we apply a convolution to produce a
        # 1-dimensional output. We do not use InstanceNorm for the first C64 layer.
        # We use leaky ReLUs with a slope of 0.2. The discriminator architecture is:
        # C64-C128-C256-C512
        self.C64 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            # We do not use InstanceNorm for the first C64 layer
            nn.LeakyReLU(0.2, True)
        )
        self.C128 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.C256 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.C512 = nn.Sequential(
            # nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),  # fanchen: stride=1, according to the original imple.
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.final_1d = nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)

    def forward(self, input):
        """Standard forward."""
        ## Your Implementation Here ##
        x = self.C64(input)
        x = self.C128(x)
        x = self.C256(x)
        x = self.C512(x)
        x = self.final_1d(x)
        return x


class AttentionDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(AttentionDiscriminator, self).__init__()
        ## Your Implementation Here ##
        self.attn_map_output = False
        self.C64 = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),
            # We do not use InstanceNorm for the first C64 layer
            nn.LeakyReLU(0.2, True)
        )
        self.C128 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True)
        )
        self.C256 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1),
            norm_layer(ndf * 4),
            nn.LeakyReLU(0.2, True)
        )
        self.attn1 = Self_Attn(ndf * 4)
        self.C512 = nn.Sequential(
            # nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1),
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1),  # fanchen: stride=1, according to the original imple.
            norm_layer(ndf * 8),
            nn.LeakyReLU(0.2, True)
        )
        self.attn2 = Self_Attn(ndf * 8)
        self.final_1d = nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)

    def forward(self, input):
        """Standard forward"""
        ## Your Implementation Here ##
        x = self.C64(input)
        x = self.C128(x)
        x = self.C256(x)
        x, attn_map1 = self.attn1(x)
        x = self.C512(x)
        x, attn_map2 = self.attn2(x)
        x = self.final_1d(x)
        if not self.attn_map_output:
            return x
        else:
            return x, attn_map1, attn_map2

