import cv2
import numpy as np
import os
from functools import lru_cache
from .core import AlignmentError, Candidate, Correspondences, Space, Transform

def phase_proposal(rgb_structure: np.ndarray, ms_structure: np.ndarray, mask: np.ndarray, max_shift_px: float) -> Transform:
    """Proposal only. Converts OpenCV's relative MS displacement into the MS->RGB correction."""
    ys, xs = np.where(mask)
    if len(xs) < 100: raise AlignmentError("ERR_INSUFFICIENT_VALID_DATA", "No common structural ROI.")
    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    a,b=rgb_structure[y0:y1,x0:x1].astype(np.float64),ms_structure[y0:y1,x0:x1].astype(np.float64)
    m=mask[y0:y1,x0:x1]
    a=np.where(m,a-a[m].mean(),0); b=np.where(m,b-b[m].mean(),0)
    shift,response=cv2.phaseCorrelate(a,b,cv2.createHanningWindow((a.shape[1],a.shape[0]),cv2.CV_64F))
    # cv2 reports where MS lies relative to RGB; correction to map MS onto RGB is its inverse.
    dx,dy=-float(shift[0]),-float(shift[1])
    if response < .15 or np.hypot(dx,dy)>max_shift_px:
        raise AlignmentError("ERR_COARSE_PROPOSAL_FAILED", f"response={response:.3f}, correction=({dx:.2f},{dy:.2f})")
    return Transform(np.array([[1.,0.,dx],[0.,1.,dy]]), Space.REGISTRATION, Space.REGISTRATION, "phase_proposal")

def tiled_sift(rgb: np.ndarray, ms: np.ndarray, mask: np.ndarray, proposal: Transform, grid=4) -> Correspondences:
    detector=cv2.SIFT_create(nfeatures=2000); matcher=cv2.BFMatcher(cv2.NORM_L2)
    h,w=rgb.shape; mss=cv2.warpAffine(ms, proposal.forward, (w,h))
    all_ms=[]; all_rgb=[]; tiles=[]
    for ty in range(grid):
      for tx in range(grid):
        y0,y1=ty*h//grid,(ty+1)*h//grid; x0,x1=tx*w//grid,(tx+1)*w//grid
        tile_mask=mask[y0:y1,x0:x1].astype(np.uint8)*255
        if tile_mask.mean()<100: continue
        kr,dr=detector.detectAndCompute(rgb[y0:y1,x0:x1],tile_mask)
        km,dm=detector.detectAndCompute(mss[y0:y1,x0:x1],tile_mask)
        if dr is None or dm is None: continue
        for pair in matcher.knnMatch(dm,dr,k=2):
          if len(pair)==2 and pair[0].distance < .70*pair[1].distance:
            pms=np.array(km[pair[0].queryIdx].pt)+[x0,y0]
            prgb=np.array(kr[pair[0].trainIdx].pt)+[x0,y0]
            # undo proposal so all model points remain in original MS registration space
            pms=proposal.inverse().apply(pms[None,:])[0]
            all_ms.append(pms); all_rgb.append(prgb); tiles.append(ty*grid+tx)
    if len(all_ms)<20: raise AlignmentError("ERR_INSUFFICIENT_MATCHES", f"Only {len(all_ms)} local matches.")
    return Correspondences(np.float32(all_ms),np.float32(all_rgb),np.array(tiles))

def _mim_descriptors(mim: np.ndarray, mask: np.ndarray, max_points: int = 180) -> tuple[np.ndarray, np.ndarray]:
    """Extract MIM grid-histogram descriptors at structural corners.

    ``mim`` contains quantised dominant Gabor orientation labels.  The descriptor
    is a 4x4 grid of local orientation histograms, with L2 normalisation.  It is
    intentionally a small, inspectable RIFT-style descriptor rather than a claim
    to implement the complete RIFT paper.
    """
    corners=cv2.goodFeaturesToTrack(mim,maxCorners=max_points,qualityLevel=.008,
        minDistance=10,mask=(mask.astype(np.uint8)*255),blockSize=5,useHarrisDetector=True)
    if corners is None: return np.empty((0,2),np.float32),np.empty((0,272),np.float32)
    # gabor_orientation_map encodes 16 orientation/scale labels in 15-value steps.
    labels=np.clip(np.rint(mim.astype(np.float32)/15.0).astype(np.int16)-1,0,16)
    points=[]; descriptors=[]
    h,w=mim.shape
    for x,y in corners.reshape(-1,2):
      x=int(round(x)); y=int(round(y)); radius=16
      if x-radius<0 or y-radius<0 or x+radius>=w or y+radius>=h: continue
      if not mask[y-radius:y+radius,x-radius:x+radius].all(): continue
      patch=labels[y-radius:y+radius,x-radius:x+radius]
      hist=[]
      for cy in range(4):
        for cx in range(4):
          cell=patch[cy*8:(cy+1)*8,cx*8:(cx+1)*8]
          hist.extend(np.bincount(cell.ravel(),minlength=17))
      descriptor=np.asarray(hist,np.float32)
      norm=np.linalg.norm(descriptor)
      if norm <= 1e-6: continue
      points.append((x,y)); descriptors.append(descriptor/norm)
    if not points: return np.empty((0,2),np.float32),np.empty((0,272),np.float32)
    return np.asarray(points,np.float32),np.asarray(descriptors,np.float32)

def tiled_mim_descriptors(rgb_mim: np.ndarray, ms_mim: np.ndarray, mask: np.ndarray, proposal: Transform, grid=4) -> Correspondences:
    """Collect automatic, tile-constrained MIM correspondences after coarse placement."""
    h,w=rgb_mim.shape
    warped_ms=cv2.warpAffine(ms_mim,proposal.forward[:2],(w,h),flags=cv2.INTER_NEAREST)
    matcher=cv2.BFMatcher(cv2.NORM_L2)
    all_ms=[]; all_rgb=[]; tiles=[]
    for ty in range(grid):
      for tx in range(grid):
        y0,y1=ty*h//grid,(ty+1)*h//grid; x0,x1=tx*w//grid,(tx+1)*w//grid
        tile_mask=mask[y0:y1,x0:x1]
        if tile_mask.mean()<.75: continue
        rp,rd=_mim_descriptors(rgb_mim[y0:y1,x0:x1],tile_mask)
        mp,md=_mim_descriptors(warped_ms[y0:y1,x0:x1],tile_mask)
        if len(rd)<2 or len(md)<2: continue
        forward={m.queryIdx:m.trainIdx for pair in matcher.knnMatch(md,rd,k=2)
                 if len(pair)==2 for m,n in [pair] if m.distance < .72*n.distance}
        reverse={m.queryIdx:m.trainIdx for pair in matcher.knnMatch(rd,md,k=2)
                 if len(pair)==2 for m,n in [pair] if m.distance < .72*n.distance}
        for mi,ri in forward.items():
          if reverse.get(ri)!=mi: continue
          pms=proposal.inverse().apply((mp[mi]+[x0,y0])[None,:])[0]
          all_ms.append(pms); all_rgb.append(rp[ri]+[x0,y0]); tiles.append(ty*grid+tx)
    if len(all_ms)<20: raise AlignmentError("ERR_INSUFFICIENT_MATCHES",f"Only {len(all_ms)} MIM correspondences.")
    return Correspondences(np.float32(all_ms),np.float32(all_rgb),np.asarray(tiles))

@lru_cache(maxsize=1)
def _loftr_model():
    """Load the official pretrained matcher lazily, so classical paths stay usable."""
    # Some Windows Python installations do not pass their current CA bundle to
    # urllib/torch.hub. Keep certificate verification enabled while supplying it.
    import certifi
    os.environ.setdefault("SSL_CERT_FILE",certifi.where())
    import torch
    from kornia.feature import LoFTR
    model=LoFTR(pretrained="outdoor")
    model.eval()
    return model

def loftr_correspondences(rgb: np.ndarray, ms: np.ndarray, mask: np.ndarray, proposal: Transform, grid: int = 7,
                          minimum_confidence: float = .50, max_dimension: int = 1024) -> Correspondences:
    """Automatic phase-prior tiled LoFTR evidence, converted back from phase-proposal to MS coordinates.

    The pretrained network is a proposal generator only. Confidence filtering,
    valid-mask filtering, distributed RANSAC and held-out QA remain mandatory.
    """
    try:
        import torch
        model = _loftr_model()
    except Exception as exc:
        raise AlignmentError("ERR_LOFTR_UNAVAILABLE", f"Pretrained LoFTR unavailable: {type(exc).__name__}") from exc
    h, w = rgb.shape
    warped_ms = cv2.warpAffine(ms, proposal.forward[:2], (w, h), flags=cv2.INTER_LINEAR)
    tiles = select_tiles(rgb, mask, grid=grid)
    if not tiles:
        raise AlignmentError("ERR_INSUFFICIENT_MATCHES", "No suitable tiles selected for LoFTR matching.")

    all_ms = []
    all_rgb = []
    tile_ids = []
    radius = 32

    for tile_id, x0, y0, x1, y1 in tiles:
        px0, py0 = max(0, x0 - radius), max(0, y0 - radius)
        px1, py1 = min(w, x1 + radius), min(h, y1 + radius)
        patch_w, patch_h = px1 - px0, py1 - py0
        if patch_w < 16 or patch_h < 16:
            continue

        run_w = max(8, int(round(patch_w)) // 8 * 8)
        run_h = max(8, int(round(patch_h)) // 8 * 8)

        rgb_patch = cv2.resize(rgb[py0:py1, px0:px1], (run_w, run_h), interpolation=cv2.INTER_AREA)
        ms_patch = cv2.resize(warped_ms[py0:py1, px0:px1], (run_w, run_h), interpolation=cv2.INTER_AREA)

        image0 = torch.from_numpy(np.ascontiguousarray(rgb_patch)).float()[None, None] / 255.0
        image1 = torch.from_numpy(np.ascontiguousarray(ms_patch)).float()[None, None] / 255.0

        try:
            with torch.inference_mode():
                output = model({"image0": image0, "image1": image1})
        except Exception as exc:
            continue

        if "keypoints0" not in output or len(output["keypoints0"]) == 0:
            continue

        kpts0 = output["keypoints0"].detach().cpu().numpy() * np.array([patch_w / run_w, patch_h / run_h], dtype=np.float32)
        kpts1 = output["keypoints1"].detach().cpu().numpy() * np.array([patch_w / run_w, patch_h / run_h], dtype=np.float32)
        confidence = output["confidence"].detach().cpu().numpy()

        tile_candidates = []
        for i, (a, b) in enumerate(zip(kpts0, kpts1)):
            if confidence[i] < minimum_confidence:
                continue
            prgb = a + np.array([px0, py0], dtype=np.float32)
            pwarped = b + np.array([px0, py0], dtype=np.float32)
            ax, ay = int(round(prgb[0])), int(round(prgb[1]))
            bx, by = int(round(pwarped[0])), int(round(pwarped[1]))

            if not (0 <= ax < w and 0 <= ay < h and 0 <= bx < w and 0 <= by < h):
                continue
            if not (mask[ay, ax] and mask[by, bx]):
                continue
            if np.linalg.norm(prgb - pwarped) > radius * 2:
                continue

            tile_candidates.append((confidence[i], prgb, pwarped))

        if not tile_candidates:
            continue

        tile_candidates.sort(key=lambda item: item[0], reverse=True)
        tile_candidates = tile_candidates[:30]

        for conf, prgb, pwarped in tile_candidates:
            pms = proposal.inverse().apply(pwarped[None, :])[0]
            all_ms.append(pms)
            all_rgb.append(prgb)
            tile_ids.append(tile_id)

    if len(all_ms) < 20:
        raise AlignmentError("ERR_INSUFFICIENT_MATCHES", f"Only {len(all_ms)} confident LoFTR matches across tiles.")

    return Correspondences(np.float32(all_ms), np.float32(all_rgb), np.array(tile_ids, dtype=int))


def fit_affine(c: Correspondences) -> Candidate:
    matrix, inliers=cv2.estimateAffine2D(c.ms,c.rgb,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
    if matrix is None: raise AlignmentError("ERR_TRANSFORM_OUT_OF_BOUNDS","Affine fit failed.")
    return Candidate(Transform(matrix,Space.REGISTRATION,Space.REGISTRATION,"tiled_sift_affine"),inliers.ravel().astype(bool),c)

def fit_similarity(c: Correspondences) -> Candidate:
    matrix,inliers=cv2.estimateAffinePartial2D(c.ms,c.rgb,method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=10000,confidence=.999)
    if matrix is None: raise AlignmentError("ERR_TRANSFORM_OUT_OF_BOUNDS","Similarity fit failed.")
    return Candidate(Transform(matrix,Space.REGISTRATION,Space.REGISTRATION,"local_similarity"),inliers.ravel().astype(bool),c)

def select_tiles(structure: np.ndarray, mask: np.ndarray, grid: int = 5, min_valid: float = .75) -> list[tuple[int,int,int,int,int]]:
    """Select distributed textured tiles; crop-row-like single-orientation tiles score lower."""
    h,w=structure.shape; candidates=[]
    gx=cv2.Sobel(structure,cv2.CV_32F,1,0); gy=cv2.Sobel(structure,cv2.CV_32F,0,1)
    for ty in range(grid):
      for tx in range(grid):
        y0,y1=ty*h//grid,(ty+1)*h//grid; x0,x1=tx*w//grid,(tx+1)*w//grid
        valid=mask[y0:y1,x0:x1]
        if valid.mean()<min_valid: continue
        vals=structure[y0:y1,x0:x1][valid]
        entropy=np.histogram(vals,bins=32,range=(0,256))[0]; entropy=-(entropy[entropy>0]/len(vals)*np.log(entropy[entropy>0]/len(vals))).sum()
        xx,xy,yy=(gx[y0:y1,x0:x1][valid]**2).mean(),(gx[y0:y1,x0:x1][valid]*gy[y0:y1,x0:x1][valid]).mean(),(gy[y0:y1,x0:x1][valid]**2).mean()
        eig=np.linalg.eigvalsh([[xx,xy],[xy,yy]])
        isotropy=float(eig[0]/max(eig[1],1e-6))
        score=float(vals.std()*entropy*(.25+.75*isotropy))
        candidates.append((score,ty*grid+tx,x0,y0,x1,y1))
    # Keep the strongest 60%, while retaining all available quadrants through the later coverage gate.
    candidates.sort(reverse=True)
    return [(tid,x0,y0,x1,y1) for _,tid,x0,y0,x1,y1 in candidates[:max(20,len(candidates)*3//5)]]

def _best_template_offset(template: np.ndarray, search: np.ndarray) -> tuple[float,float,float,float]:
    result=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
    _,peak,_,loc=cv2.minMaxLoc(result)
    flat=np.sort(result.ravel())
    secondary=float(flat[-2]) if flat.size>1 else -1.
    return float(loc[0]),float(loc[1]),float(peak),float(peak/max(secondary,1e-4))

def _ecc_translation(template: np.ndarray, patch: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Return patch-to-template translation from ECC, or None when multimodal refinement cannot converge."""
    try:
        a=template.astype(np.float32)/255.; b=patch.astype(np.float32)/255.
        warp=np.eye(2,3,dtype=np.float32)
        score,warp=cv2.findTransformECC(a,b,warp,cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,80,1e-5),None,1)
        if not np.isfinite(score) or score < .55: return None
        return warp[:,2].astype(float),float(score)
    except cv2.error:
        return None

def tiled_local_displacements(rgb: np.ndarray, ms: np.ndarray, mask: np.ndarray, proposal: Transform, grid: int=5, radius: int=24) -> Correspondences:
    """Automatic control points from bounded structural patch matching.

    MS is first placed near RGB by the phase proposal. Every accepted local match is
    converted back into original-MS coordinates so the fitted result remains MS->RGB.
    """
    h,w=rgb.shape; warped_ms=cv2.warpAffine(ms,proposal.forward[:2],(w,h),flags=cv2.INTER_LINEAR)
    accepted_ms=[]; accepted_rgb=[]; accepted_tile=[]
    for tile_id,x0,y0,x1,y1 in select_tiles(rgb,mask,grid):
      template=rgb[y0:y1,x0:x1]
      th,tw=template.shape
      sx0=max(0,x0-radius); sy0=max(0,y0-radius); sx1=min(w,x1+radius); sy1=min(h,y1+radius)
      search=warped_ms[sy0:sy1,sx0:sx1]
      if search.shape[0]<th or search.shape[1]<tw: continue
      ox,oy,peak,ratio=_best_template_offset(template,search)
      if peak < .25 or ratio < 1.05: continue
      candidate_top=np.array([sx0+ox,sy0+oy]); candidate_center=candidate_top+np.array([tw/2,th/2])
      rgb_center=np.array([(x0+x1)/2,(y0+y1)/2])
      # Local phase compares the candidate patch with the RGB template; it must agree with template location.
      patch=warped_ms[int(candidate_top[1]):int(candidate_top[1])+th,int(candidate_top[0]):int(candidate_top[0])+tw]
      shift,response=cv2.phaseCorrelate(template.astype(np.float64),patch.astype(np.float64))
      # Correction moving patch to template is negative relative shift.
      phase_candidate=candidate_center-np.array(shift)
      if response < .15 or np.linalg.norm(phase_candidate-rgb_center)>radius: continue
      ecc=_ecc_translation(template,patch)
      ecc_candidate=candidate_center if ecc is None else candidate_center-ecc[0]
      # Template, phase, and ECC must agree. ECC failure is tolerated only where phase is sharp.
      if np.linalg.norm(phase_candidate-candidate_center) > 3.0: continue
      if ecc is not None and np.linalg.norm(ecc_candidate-candidate_center)>3.0: continue
      # Reverse test: use the matching MS patch to recover original RGB tile location.
      reverse_search=rgb[sy0:sy1,sx0:sx1]
      rox,roy,rpeak,rratio=_best_template_offset(patch,reverse_search)
      reverse_center=np.array([sx0+rox+tw/2,sy0+roy+th/2])
      if rpeak < .25 or rratio < 1.05 or np.linalg.norm(reverse_center-rgb_center)>1.5: continue
      refined_center=(candidate_center+phase_candidate+ecc_candidate)/3 if ecc is not None else (candidate_center+phase_candidate)/2
      original_ms=proposal.inverse().apply(refined_center[None,:])[0]
      accepted_ms.append(original_ms); accepted_rgb.append(rgb_center); accepted_tile.append(tile_id)
    if len(accepted_ms)<20: raise AlignmentError("ERR_INSUFFICIENT_MATCHES",f"Only {len(accepted_ms)} confident local tile correspondences.")
    return Correspondences(np.float32(accepted_ms),np.float32(accepted_rgb),np.asarray(accepted_tile))
