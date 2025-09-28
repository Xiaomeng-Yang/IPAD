from strhub.models.utils import create_model


dependencies = ['torch', 'pytorch_lightning', 'timm']


def pimnet(pretrained: bool = False, num_iter: int = 5, **kwargs):
    """
    ABINet model (img_size=128x32)
    @param pretrained: (bool) Use pretrained weights
    @param num_iter: (int) number of refinement iterations to use
    """
    return create_model('pimnet', pretrained, num_iter=num_iter, **kwargs)


def ipad(pretrained: bool = False, num_iter: int = 5, **kwargs):
    """
    ABINet model (img_size=128x32)
    @param pretrained: (bool) Use pretrained weights
    @param num_iter: (int) number of refinement iterations to use
    """
    return create_model('ipad', pretrained, num_iter=num_iter, **kwargs)
