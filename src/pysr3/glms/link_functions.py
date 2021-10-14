from typing import Union

import numpy as np
from numpy.typing import ArrayLike


class LinkFunction:
    def value(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        raise NotImplementedError("Abstract class's method is called")

    def gradient(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        raise NotImplementedError("Abstract class's method is called")

    def hessian(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        raise NotImplementedError("Abstract class's method is called")


class IdentityLinkFunction(LinkFunction):
    def value(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return x

    def gradient(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return np.ones(len(x))

    def hessian(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return 0


class PoissonLinkFunction(LinkFunction):
    def value(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return np.exp(x)

    def gradient(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return np.exp(x)

    def hessian(self, x: Union[float, ArrayLike]) -> Union[float, ArrayLike]:
        return np.exp(x)
