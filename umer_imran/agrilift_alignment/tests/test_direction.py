import cv2
import numpy as np
import pytest
from agrilift_alignment.core import Space, Transform
from agrilift_alignment.registration import phase_proposal
from agrilift_alignment.registration import tiled_local_displacements, tiled_mim_descriptors
from agrilift_alignment.core import AlignmentError
from agrilift_alignment.preprocessing import log_map
from agrilift_alignment.preprocessing import orientation_map
from agrilift_alignment.preprocessing import gabor_orientation_map
from agrilift_alignment.preprocessing import gabor_energy_map

@pytest.mark.parametrize("dx,dy",[(7,0),(-7,0),(0,5),(0,-5),(4,-3)])
def test_phase_proposal_moves_ms_landmark_toward_rgb(dx,dy):
    rgb=np.zeros((160,160),np.uint8)
    cv2.circle(rgb,(80,80),9,255,-1); cv2.rectangle(rgb,(30,40),(45,55),180,-1)
    # MS is physically displaced relative to the RGB reference.
    ms=cv2.warpAffine(rgb,np.array([[1,0,dx],[0,1,dy]],np.float32),(160,160))
    proposal=phase_proposal(rgb,ms,np.ones_like(rgb,bool),30)
    before=np.linalg.norm(np.array([80+dx,80+dy])-np.array([80,80]))
    after=np.linalg.norm(proposal.apply(np.array([[80+dx,80+dy]],float))[0]-np.array([80,80]))
    assert after < before

def test_forward_inverse_round_trip():
    t=Transform(np.array([[1,0,5],[0,1,-3]],float),Space.MS_SOURCE,Space.RGB_REFERENCE,"test")
    p=np.array([[10.,20.]])
    assert np.allclose(t.inverse().apply(t.apply(p)),p)

def test_local_tiles_create_automatic_correspondences():
    rng=np.random.default_rng(3)
    rgb=(rng.random((400,400))*255).astype(np.uint8)
    for x,y in [(70,80),(160,250),(300,120),(280,310),(110,330)]:
        cv2.circle(rgb,(x,y),14,255,-1)
    ms=cv2.warpAffine(rgb,np.array([[1,0,6],[0,1,-4]],np.float32),(400,400))
    mask=np.ones_like(rgb,bool)
    proposal=phase_proposal(rgb,ms,mask,30)
    c=tiled_local_displacements(rgb,ms,mask,proposal,grid=5,radius=20)
    assert len(c.ms) >= 20

def test_log_map_excludes_invalid_boundary():
    yy,xx=np.mgrid[:100,:100]
    image=(.03+.04*((xx//8+yy//8)%2)).astype(np.float32); image[:10]=2**32
    mask=np.ones((100,100),bool); mask[:10]=False
    result=log_map(image,mask)
    assert result[:10].max()==0

def test_orientation_map_excludes_invalid_boundary():
    yy,xx=np.mgrid[:100,:100]
    image=(.03+.04*((xx//8+yy//8)%2)).astype(np.float32); image[:10]=2**32
    mask=np.ones((100,100),bool); mask[:10]=False
    assert orientation_map(image,mask)[:10].max()==0

def test_gabor_orientation_map_excludes_invalid_boundary():
    yy,xx=np.mgrid[:100,:100]
    image=(.03+.04*((xx//8+yy//8)%2)).astype(np.float32); image[:10]=2**32
    mask=np.ones((100,100),bool); mask[:10]=False
    result=gabor_orientation_map(image,mask)
    assert result[:10].max()==0
    assert result[20:].max()>0

def test_gabor_energy_map_excludes_invalid_boundary():
    yy,xx=np.mgrid[:100,:100]
    image=(.03+.04*((xx//8+yy//8)%2)).astype(np.float32); image[:10]=2**32
    mask=np.ones((100,100),bool); mask[:10]=False
    result=gabor_energy_map(image,mask)
    assert result[:10].max()==0
    assert result[20:].max()>0

def test_mim_tiles_create_automatic_correspondences():
    rng=np.random.default_rng(22)
    source=(rng.random((400,400))*255).astype(np.uint8)
    for x,y in [(70,80),(160,250),(300,120),(280,310),(110,330)]:
        cv2.circle(source,(x,y),14,255,-1)
    ms=cv2.warpAffine(source,np.array([[1,0,6],[0,1,-4]],np.float32),(400,400))
    mask=np.ones_like(source,bool)
    proposal=phase_proposal(source,ms,mask,30)
    c=tiled_mim_descriptors(gabor_orientation_map(source,mask),gabor_orientation_map(ms,mask),mask,proposal,grid=5)
    assert len(c.ms)>=20

