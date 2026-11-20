"""Associate saved wide-crop detections with an original clipped display line.

This is uncertain cleanup geometry, never corrected OCR, a translation selection,
or a semantic stroke mask. The caller protects previous output and other sources.
"""
import unicodedata

import cv2
import numpy as np
from PIL import Image,ImageDraw

from . import natural as n, preparation as p
from ..ocr.source_localization import extent_crop_bounds


POLICY='crop_corroborated_display_extent_v1'


def normalized(text):
    return ''.join(unicodedata.normalize('NFKC',text).split())


def derive(rgb,raw,unit,record):
    """Require one aligned, uniquely source-associated continuation prediction."""
    bounds=extent_crop_bounds(raw,unit.member_ids,rgb.size)
    n.require(record.get('crop_kind')=='extent' and record['bounds']==bounds
        and record['unit_id']==unit.unit_id and record['unit_revision']==unit.revision
        and n.canonical(record['member_ids'])==n.canonical(unit.member_ids), 'extent source binding drift')
    crop=rgb.crop(bounds)
    n.require(record['crop_pixels_sha256']==n.digest(crop.tobytes()),'extent crop pixels drift')
    prediction=record['prediction']
    n.require(record['prediction_revision']==n.revision(prediction),'extent prediction drift')
    fields=('rec_texts','rec_scores','rec_polys')
    n.require(all(type(prediction.get(k)) is list for k in fields) and
        len({len(prediction[k]) for k in fields})==1 and len(prediction['rec_texts'])<=64,'extent prediction association')
    selected={(type(i).__name__,i) for i in unit.member_ids}
    member=next(d for d in raw if (type(d['source_id']).__name__,d['source_id']) in selected)
    x,y,r,b=bounds
    local=[[a-x,c-y] for a,c in member['polygon']]
    own=np.asarray(n.polygon_mask(local,crop.size,evidence=True))!=0
    box=cv2.boxPoints(cv2.minAreaRect(np.asarray(local,dtype=np.float32)))
    edges=np.roll(box,-1,axis=0)-box;lengths=np.linalg.norm(edges,axis=1)
    axis=edges[int(lengths.argmax())]/lengths.max();cross=np.array([-axis[1],axis[0]])
    original_points=np.asarray(local);along=original_points@axis;across=original_points@cross
    original=normalized(unit.selected.text);candidates=[]
    n.require(original,'empty extent source reading')
    for index,(text,score,poly) in enumerate(zip(*(prediction[k] for k in fields))):
        n.require(type(text) is str and type(score) in (int,float) and np.isfinite(score) and 0<=score<=1,
            'extent reading/score contract')
        mask=np.asarray(n.polygon_mask(poly,crop.size,evidence=True))!=0
        reading=normalized(text)
        if score<.7 or not (reading==original or (len(original)>=2 and
                (reading.startswith(original) or reading.endswith(original)))):continue
        overlap=int((mask&own).sum());area=int(mask.sum())
        if overlap<.8*int(own.sum()) or overlap<.35*area or area>3*int(own.sum()):continue
        if int((mask&~p.dilate(own,12)).sum())<16:continue
        points=np.asarray(poly,dtype=np.float64);major=points@axis;minor=points@cross
        thickness=float(np.ptp(across))+1
        if not (.65*thickness<=np.ptp(minor)+1<=1.5*thickness and
                abs((minor.max()+minor.min()-across.max()-across.min())/2)<=.35*thickness and
                np.ptp(major)+1<=3*(np.ptp(along)+1)):continue
        # A crop-boundary box may still be cut off. Do not promote its edge into
        # a supposed full extent, or merge a neighboring source into this line.
        if not (points.min(axis=0)>=3).all() or not (points.max(axis=0)<np.array(crop.size)-3).all():continue
        combined=np.concatenate((points,original_points))
        if max(np.ceil(combined.max(axis=0))-np.floor(combined.min(axis=0))+1)>512:continue
        conflict=False
        for d in raw:
            key=(type(d['source_id']).__name__,d['source_id'])
            if key in selected:continue
            neighbor_image=Image.new('L',crop.size)
            ImageDraw.Draw(neighbor_image).polygon([(a-x,c-y) for a,c in d['polygon']],fill=255)
            neighbor=np.asarray(neighbor_image)!=0
            neighbor_area=abs(cv2.contourArea(np.asarray(d['polygon'],dtype=np.float32)))
            if neighbor.any() and np.count_nonzero(neighbor&mask&~own)>=.5*max(1,neighbor_area):
                conflict=True;break
        if conflict:continue
        candidates.append((index,text,score,[[a+x,c+y] for a,c in poly],overlap,area))
    n.require(len(candidates)==1,'no unique aligned source-continuation extent')
    index,text,score,polygon,overlap,area=candidates[0]
    evidence={'policy':POLICY,'prediction_index':index,'prediction_revision':record['prediction_revision'],
        'crop_bounds':bounds,'crop_pixels_sha256':record['crop_pixels_sha256'],'polygon':polygon,
        'original_source_text':unit.selected.text,'observed_text':text,'score':score,
        'source_overlap_pixels':overlap,'predicted_polygon_pixels':area,
        'reading_changed':text!=unit.selected.text,'reading_promoted':False,'translation_reselected':False,
        'raw_geometry_unchanged':True,'semantic_mask':False,'source_removal_verified':False,
        'possible_collateral_art_loss':True}
    return {**evidence,'revision':n.revision(evidence)}


def raster(extent,bounds):
    x,y,r,b=bounds;mask=Image.new('L',(r-x,b-y))
    ImageDraw.Draw(mask).polygon([(a-x,c-y) for a,c in extent['polygon']],fill=255)
    return np.asarray(mask)!=0


def apply(mask,extent,bounds,protected):
    """A corroborated extent is a full cleanup hypothesis, not just extra edges."""
    return (mask|p.dilate(raster(extent,bounds),2))&~protected
