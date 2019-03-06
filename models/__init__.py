from functools import partial

from .fc import FC
from .quadra_fc import QuadraFC
from .energy_based import EB
from .oc_svm import OCSVM


MODELS = {
    'fc': partial(FC, use_batchnorm=False),
    'fc_bn': partial(FC, use_batchnorm=True),
    'quadra_fc': partial(QuadraFC, use_batchnorm=False),
    'quadra_fc_bn': partial(QuadraFC, use_batchnorm=True),
    'eb': EB,
    'oc_svm': OCSVM,
}
