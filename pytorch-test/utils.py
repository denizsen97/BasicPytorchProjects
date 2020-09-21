import torch



class toYCbCr(object):
    """
    Convert a RGB image to YCbCr
    """

    def __call__(self, pic):
        pic = 256 * pic
        R = pic[0, :, :]
        G = pic[1, :, :]
        B = pic[2, :, :]
        # input is mini-batch N x 3 x H x W of an RGB image
        out = torch.zeros_like(pic)
        out[0, :, :] =  0.299 * R + 0.587 * G + 0.114 * B

        out[1, :, :] =  128 + -0.168736 * R - 0.331264 * G + 0.5 * B
        
        out[2, :, :] =  128 + 0.5 * R - 0.418688 * G - 0.081312 * B
        
        

        return out

    def __repr__(self):
        return self.__class__.__name__ +    '()'

