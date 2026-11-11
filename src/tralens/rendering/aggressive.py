"""Crop-local removal after precise routes, before the independent visibility pass.

Art contact is a soft risk. Foreign bindings, overwritten committed pixels and
target text loss are hard errors. Geometry masks are never called semantic masks.
"""
from copy import deepcopy
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw

from . import natural as n, preparation as p, typography
from . import reconstruction as rec


POLICY='crop_local_aggressive_v2'
MAX_REGIONS=64
MAX_LOCAL_WORK=16_777_216
HARD_REASONS={'runtime_locality','alpha_containment','erasure_obstacle','typography_containment',
    'evaluation_cap','replacement_unchanged_text'}


def settings(removal,backend,config):
    n.require(removal in ('legacy','preserve','aggressive'),'unknown removal policy')
    n.require(backend in ('classical','lama'),'unknown reconstruction policy')
    n.require(backend!='lama' or (removal=='aggressive' and config is not None),
        'LaMa requires aggressive removal and explicit approved local reconstruction configuration')
    n.require(config is None or backend=='lama','reconstruction configuration supplied while backend disabled')


class Geometry:
    """Cache bounds and a spatial grid; rasterize only intersecting local polygons."""
    def __init__(self,raw,size):
        self.size=size;self.raw={};self.boxes={};self.grid={}
        for d in raw:
            key=(type(d['source_id']).__name__,d['source_id'])
            rec.require(key not in self.raw,'duplicate raw source ID')
            points=np.asarray(d['polygon']);rec.require(points.ndim==2 and points.shape[1]==2 and np.isfinite(points).all(),
                'invalid raw source polygon')
            x,y=map(int,np.floor(points.min(axis=0)));r,b=map(int,np.ceil(points.max(axis=0))+1)
            rec.require(0<=x<r<=size[0]+1 and 0<=y<b<=size[1]+1,'source polygon outside page')
            self.raw[key]=d;self.boxes[key]=(x,y,min(size[0],r),min(size[1],b))
            for tx in range(x//128,(r-1)//128+1):
                for ty in range(y//128,(b-1)//128+1):self.grid.setdefault((tx,ty),set()).add(key)

    def local(self,member_ids,reserved,margin=48):
        keys={(type(i).__name__,i) for i in member_ids}
        rec.require(keys and keys<=self.raw.keys() and len(keys)==len(member_ids),'foreign/duplicate recovery membership')
        boxes=[self.boxes[k] for k in keys]
        x=max(0,min(b[0] for b in boxes)-margin);y=max(0,min(b[1] for b in boxes)-margin)
        r=min(self.size[0],max(b[2] for b in boxes)+margin);b=min(self.size[1],max(b[3] for b in boxes)+margin)
        n.budget_check(max(r-x,b-y)<=512,'native reconstruction crop exceeds fixed 512 model/context scope')
        nearby=set()
        for tx in range(x//128,(r-1)//128+1):
            for ty in range(y//128,(b-1)//128+1):nearby.update(self.grid.get((tx,ty),()))
        own=Image.new('L',(r-x,b-y));other=own.copy()
        for key in nearby:
            bx,by,br,bb=self.boxes[key]
            if br<=x or bb<=y or bx>=r or by>=b:continue
            target=own if key in keys else other
            ImageDraw.Draw(target).polygon([(px-x,py-y) for px,py in self.raw[key]['polygon']],fill=255)
        protected=p.dilate(np.asarray(other)!=0,2)|reserved[y:b,x:r]
        return [x,y,r,b],np.asarray(own)!=0,protected


def masks(crop,own,protected,*,probability=None):
    """Return distinct evidence and removal candidates; uncertain cuts are allowed."""
    # Multichannel local contrast captures light outlines and colored lettering;
    # it is only image evidence, NOT a learned semantic prediction.
    smooth=cv2.GaussianBlur(crop,(5,5),0)
    contrast=np.max(np.abs(crop.astype(np.int16)-smooth.astype(np.int16)),axis=2)>12
    contrast &= p.dilate(own,4)
    candidates=[]
    if probability is not None:
        rec.require(probability.shape==own.shape and probability.dtype==np.uint8,'local learned mask shape/type')
        support=(probability>=77)&p.dilate(own,6)
        candidates.append(('learned_support_expanded',p.dilate(support,2)&~protected))
        candidates.append(('learned_support_plus_source_geometry',(p.dilate(support,3)|p.dilate(own,1))&~protected))
    candidates.extend([('padded_source_polygons',p.dilate(own,1)&~protected),
        ('expanded_source_polygons',p.dilate(own,5)&~protected)])
    return contrast,candidates,{'predicted_lettering':'learned_probability' if probability is not None else 'unavailable',
        'native_evidence':'multichannel_local_contrast_only','native_component_route':'attempted separately before recovery',
        'omitted_ladder_rungs':[] if probability is not None else ['learned_glyph_support','learned_support_plus_geometry'],
        'foreground_is_text_certified':False,'possible_collateral_art_loss':True,
        'bounded_cut_across_art_contact':True,'source_pixels_protected':int((own&protected).sum())}


def assess(original,candidate,mask,evidence,protected):
    """Residual/seam/edge observations, not a learned quality or truth score."""
    delta=np.max(np.abs(candidate.astype(np.int16)-original.astype(np.int16)),axis=2)
    remaining=evidence & ~protected & (delta<=3)
    outside=p.dilate(mask,3)&~mask&~protected
    edge=cv2.Canny(cv2.cvtColor(candidate,cv2.COLOR_RGB2GRAY),60,150)>0
    before_edge=cv2.Canny(cv2.cvtColor(original,cv2.COLOR_RGB2GRAY),60,150)>0
    count,_,stats,_=cv2.connectedComponentsWithStats((edge&mask).astype(np.uint8),8)
    text_like=sum(3<=int(s[4])<=180 and 2<=int(s[2])<=48 and 2<=int(s[3])<=48 for s in stats[1:])
    ring=mask & p.dilate(outside,1)
    jump=float(np.abs(cv2.Laplacian(candidate.astype(np.float32),cv2.CV_32F)).mean(axis=2)[ring].mean()) if ring.any() else 0
    return {'unchanged_source_contrast_pixels':int(remaining.sum()),'evidence_pixels':int((evidence&~protected).sum()),
        'outer_residual_pixels':int((remaining&~mask).sum()),'seam_laplacian_mean':jump,
        'boundary_edge_count_before':int((before_edge&outside).sum()),'boundary_edge_count_after':int((edge&outside).sum()),
        'new_text_like_components':int(text_like),'text_like_components_are_semantic_ocr':False,
        'ocr_silence_used':False,'quality_verified':False}


def cheap(crop,mask,protected):
    ring=p.dilate(mask,4)&~p.dilate(mask,1)&~protected
    samples=crop[ring]
    if len(samples)>=32:
        color=np.median(samples,axis=0).astype(np.uint8)
        if np.max(np.abs(samples.astype(np.int16)-color))<=3:
            candidate=crop.copy();candidate[mask]=color
            return candidate,'measured_uniform_fill'
    candidate=cv2.inpaint(crop,mask.astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
    return candidate,'classical_telea_unverified'


def recover(rgb,source,raw,rows,clean,text,erase,layout,font_data,coverage,style,*,backend=None,budget=None):
    """Independent bounded pass; native successes are retained, then visibility runs.

    Every region gets one cheap attempt before any retry. Learned quota is separate
    and never consumes baseline/display budget. Per-region work is local, not page².
    """
    started=time.monotonic();budget=budget or rec.Budget();geometry=Geometry(raw,rgb.size)
    original=np.asarray(rgb);plate=np.asarray(clean).copy();new_text=text.copy()
    erase=erase.copy();layout=layout.copy();rows=deepcopy(rows)
    alpha=np.asarray(text.getchannel('A'))!=0
    reserved=p.dilate(erase|alpha,2)
    initial_reserved=reserved.copy();native_plate=plate.copy();native_text=np.asarray(text).copy()
    summary={'policy':POLICY,'attempted':0,'learned_calls':0,'learned_failures':0,'retries':0,'local_work_pixels':0,
        'budget_exclusions':0,'selected_learned':0,'selected_classical':0,'selected_fill':0,
        'model_config':getattr(backend,'config',None),'limits':{'regions':MAX_REGIONS,'learned_calls':budget.maximum_calls,
        'learned_seconds':budget.maximum_seconds,'local_work_pixels':MAX_LOCAL_WORK},'artwork_recovered':False}
    artifacts={};jobs=[];backend_error=None
    members={(type(m).__name__,m) for row in rows if row['status']=='automatic_composed'
        for m in row.get('binding',{}).get('member_ids',[])}
    for i,row in enumerate(rows):
        if row['status']=='automatic_composed' or 'binding' not in row or row.get('reason_code') in HARD_REASONS:continue
        entry={'status':'unresolved','prior_reason':row.get('reason_code'),'attempts':[]};row['reconstruction']=entry
        membership={(type(m).__name__,m) for m in row['binding']['member_ids']}
        if membership & members:
            entry['reason']='membership_already_displayed';continue
        if len(jobs)>=MAX_REGIONS:
            entry['reason']='region_budget';summary['budget_exclusions']+=1;continue
        try:
            bounds,own,protected=geometry.local(row['binding']['member_ids'],reserved)
            x,y,r,b=bounds;crop=original[y:b,x:r].copy()
            rec.require(crop.size and own.any(),'empty recovery source')
            # Keep fonts/wording fixed before any destructive work/model dispatch.
            # Presentation space is independent of the larger inference context.
            allowed=p.dilate(own,24)&~protected
            n.validate_text(row['binding']['text'],coverage)
            layer,proof,content=typography.layout_dialogue(row['binding']['text'],font_data,coverage,
                Image.fromarray(allowed.astype(np.uint8)*255),{**style,'fill':[0,0,0],'outline':1,'outline_fill':[255,255,255]})
            rec.require(proof['original_text']==row['binding']['text'] and proof['size']>=style['min_size'],
                'reconstruction layout dropped wording or reduced requested floor')
            predicted,ladder,evidence=masks(crop,own,protected)
            entry.update(bounds=bounds,layout=proof,mask_evidence=evidence)
            artifacts[f'recovery-{i:03d}-original.png']=Image.fromarray(crop)
            artifacts[f'recovery-{i:03d}-lettering-evidence.png']=Image.fromarray(predicted.astype(np.uint8)*255)
            artifacts[f'recovery-{i:03d}-layout.png']=content
            jobs.append({'i':i,'bounds':bounds,'crop':crop,'protected':protected,'predicted':predicted,
                'ladder':ladder,'layer':layer,'proof':proof,'best':None,'retry':False})
        except rec.BindingError:raise
        except (ValueError,OSError,cv2.error) as error:entry.update(reason=str(error),status='layout_or_crop_unavailable')
    for round_index in range(2):
        for job in jobs:
            if round_index and not job['retry']:continue
            i=job['i'];row=rows[i];entry=row['reconstruction'];crop=job['crop'];protected=job['protected']
            if summary['local_work_pixels']+crop.shape[0]*crop.shape[1]>MAX_LOCAL_WORK:
                entry['reason']='local_work_budget';summary['budget_exclusions']+=1;continue
            summary['local_work_pixels']+=crop.shape[0]*crop.shape[1]
            if round_index:summary['retries']+=1
            else:summary['attempted']+=1
            name,mask=job['ladder'][round_index]
            if not mask.any():entry['reason']='protected_source_overlap';continue
            req=rec.request(source,row['binding'],job['bounds'],crop,mask,mask,protected,attempt=round_index,
                model={k:backend.config.get(k) for k in ('backend','sha256','config_revision')} if backend else None)
            prefix=f'recovery-{i:03d}-{round_index}'
            artifacts[prefix+'-request.json']=rec.canonical(req).encode()
            artifacts[prefix+'-inference-mask.png']=Image.fromarray(mask.astype(np.uint8)*255)
            artifacts[prefix+'-commit-mask.png']=Image.fromarray(mask.astype(np.uint8)*255)
            candidate,method=cheap(crop,mask,protected)
            attempt={'mask':name,'request_revision':req['revision'],'method':method,'status':'candidate'}
            cheap_signals=assess(crop,candidate,mask,job['predicted'],protected)
            defer_learned=round_index==0 and cheap_signals['outer_residual_pixels']>0
            if method!='measured_uniform_fill' and backend is not None:
                if defer_learned:attempt['learned_skipped']='deferred_until_residual_mask_expansion'
                elif backend_error:attempt['learned_skipped']='backend_circuit_open: '+backend_error
                elif not budget.available():attempt['learned_skipped']='separate_learned_budget';summary['budget_exclusions']+=1
                else:
                    budget.calls+=1;clock=time.monotonic()
                    try:
                        result=backend.reconstruct(req,crop,mask,mask,protected,
                            timeout=min(30,max(.1,budget.maximum_seconds-budget.seconds)))
                        candidate=rec.compose(req,crop,mask,mask,protected,result)
                        method='learned_lama';attempt['method']=method
                        attempt['backend_result']={k:v for k,v in result.items() if k!='pixels'}
                    except rec.BindingError:raise
                    except Exception as error:
                        backend_error=type(error).__name__+': '+str(error)[:1000]
                        attempt['learned_failure']=backend_error;summary['learned_failures']+=1
                    finally:budget.seconds+=time.monotonic()-clock
            # Even classical/native output is explicitly composited through commit.
            candidate=rec.compose(req,crop,mask,mask,protected,{'request_revision':req['revision'],'pixels':candidate})
            signals=assess(crop,candidate,mask,job['predicted'],protected)
            attempt['assessment']=signals;entry['attempts'].append(attempt)
            artifacts[prefix+'-cleanplate.png']=Image.fromarray(candidate)
            priority={'measured_uniform_fill':0,'learned_lama':1,'classical_telea_unverified':2}[method]
            # Residual evidence precedes footprint cost. Do not optimize smoothness
            # or claim an aesthetic oracle; a valid learned result precedes cheap
            # classical output on equally complete masks.
            rank=(signals['outer_residual_pixels'],priority,int(mask.sum()))
            if job['best'] is None or rank<job['best'][0]:job['best']=(rank,candidate,mask,method,round_index,signals)
            job['retry']=round_index==0 and signals['outer_residual_pixels']>0
    for job in jobs:
        if job['best'] is None:continue
        i=job['i'];row=rows[i];entry=row['reconstruction']
        _,candidate,mask,method,chosen,signals=job['best'];x,y,r,b=job['bounds'];layer=job['layer']
        local_alpha=np.asarray(layer.getchannel('A'))!=0
        membership={(type(m).__name__,m) for m in row['binding']['member_ids']}
        # Reservations may have grown since layout planning. Reject atomically;
        # neither old English nor earlier cleanup can be overwritten by a later job.
        if membership & members or ((mask|local_alpha)&reserved[y:b,x:r]).any():
            entry.update(status='collision_retained_primary',reason='earlier_output_reserved');continue
        plate[y:b,x:r][mask]=candidate[mask];erase[y:b,x:r]|=mask;layout[y:b,x:r]|=local_alpha
        new_text.alpha_composite(layer,dest=(x,y));alpha[y:b,x:r]|=local_alpha
        occupied=mask|local_alpha
        reserved[y:b,x:r]|=p.dilate(occupied,2)
        members.update(membership)
        entry.update(status='composed_unverified',selected_attempt=chosen,method=method,assessment=signals,
            source_removal_verified=False,artwork_loss_possible=True)
        row.update(status='automatic_composed',reason_code=None,reason=None,render_quality='best_effort',
            cleanup_method=method,layout={**job['proof'],'crop_origin':[x,y]},
            best_effort={'warnings':['Bounded aggressive removal may damage local artwork; reconstructed pixels are speculative.']})
        artifacts[f'recovery-{i:03d}-text.png']=layer
        summary[{'learned_lama':'selected_learned','classical_telea_unverified':'selected_classical',
            'measured_uniform_fill':'selected_fill'}[method]]+=1
    final_clean=Image.fromarray(plate);composite=Image.alpha_composite(final_clean.convert('RGBA'),new_text).convert('RGB')
    rec.require(np.array_equal(plate[initial_reserved],native_plate[initial_reserved]),'recovery overwrote committed cleanup')
    rec.require(np.array_equal(np.asarray(new_text)[initial_reserved],native_text[initial_reserved]),'recovery overwrote accepted English')
    n.verify_pixel_audit(n.pixel_audit(rgb,final_clean,composite,erase,alpha,~(erase|layout)))
    summary.update(learned_calls=budget.calls,learned_seconds=budget.seconds,seconds=time.monotonic()-started)
    return rows,final_clean,new_text,composite,erase,layout,artifacts,summary
