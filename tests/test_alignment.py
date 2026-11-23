import numpy as np
import pytest
from visage.alignment import similarity_transform


def test_known_rotation_scale_translation_are_recovered_without_shear():
    source=np.array([[0,0],[1,0],[0,1],[2,3]],dtype=float)
    theta=.35
    matrix=2*np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    target=source@matrix.T+[5,8]
    actual=similarity_transform(source,target)
    np.testing.assert_allclose(np.c_[source,np.ones(4)]@actual.T,target,atol=1e-12)
    assert np.linalg.det(actual[:,:2])>0
    np.testing.assert_allclose(actual[:,:2]@actual[:,:2].T,np.eye(2)*4,atol=1e-12)


def test_degenerate_nonfinite_and_wrong_shape_landmarks_fail():
    for source,target in (([[1,1],[1,1]],[[2,3],[4,5]]),([[0,0],[1,1]],[[1,1],[1,1]]),
                          ([[0,np.nan],[1,1]],[[0,0],[1,1]]),([1,2],[3,4])):
        with pytest.raises(ValueError):
            similarity_transform(source,target)
