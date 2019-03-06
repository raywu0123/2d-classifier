from .gaussian import AffineGaussians
from .sklearn_datasets import (
    SklearnMoons,
    SklearnCircles,
    SklearnBlobs,
    SklearnClassification,
)
from .angular import Angular

data_providers = {
    'AffineGaussian2': AffineGaussians(num_class=2),
    'AffineGaussian3': AffineGaussians(num_class=3),

    'sklearn_moons': SklearnMoons(),
    'sklearn_circles': SklearnCircles(),

    'sklearn_blobs2': SklearnBlobs(num_class=2),
    'sklearn_blobs3': SklearnBlobs(num_class=3),
    'sklearn_blobs4': SklearnBlobs(num_class=4),
    'sklearn_blobs10': SklearnBlobs(num_class=10),

    'sklearn_classification2': SklearnClassification(num_class=2),
    'sklearn_classification3': SklearnClassification(num_class=3),
    'sklearn_classification4': SklearnClassification(num_class=4),
    'sklearn_classification10': SklearnClassification(num_class=10),

    'angular2': Angular(num_class=2),
    'angular3': Angular(num_class=3),
    'angular4': Angular(num_class=4),
    'angular10': Angular(num_class=10),
}
