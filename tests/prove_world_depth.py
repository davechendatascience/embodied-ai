import os, sys, numpy as np, imageio
REPO="/home/edge-host/Documents/GitHub/embodied_ai"
sys.path.insert(0, REPO+"/src")
from rekep_libero import add_rekep_to_path; add_rekep_to_path()
from rekep_libero.config import load_config
from rekep_libero.environment_libero import ReKepLiberoEnv, AGENTVIEW, CAM_NAMES
from rekep_libero.world_depth import perceived_world
from rekep_libero.world_export import export
cfg=load_config(); ec=dict(cfg["env"])
ec["bounds_min"],ec["bounds_max"]=cfg["workspace"]["bounds_min"],cfg["workspace"]["bounds_max"]
ec["interpolate_pos_step_size"]=cfg["main"]["interpolate_pos_step_size"]
ec["interpolate_rot_step_size"]=cfg["main"]["interpolate_rot_step_size"]
env=ReKepLiberoEnv(ec,task_suite="libero_goal",task_id=0,robot="Panda",
                   resolution=cfg["libero"]["resolution"])
def overlay(world, colour):
    """Project obstacle centres into agentview and mark them on the RGB."""
    rgb=np.asarray(env._last_obs[f"{CAM_NAMES[AGENTVIEW]}_image"])[::-1].copy()
    K,cam2world=env._camera_geometry(CAM_NAMES[AGENTVIEW])
    w2c=np.linalg.inv(cam2world)
    C=np.array([s["pose"][:3] for s in world["cuboid"].values()])
    cam=(np.c_[C,np.ones(len(C))]@w2c.T)[:,:3]
    ok=cam[:,2]>1e-6; cam=cam[ok]
    u=(cam[:,0]/cam[:,2]*K[0,0]+K[0,2]).astype(int)
    v=(cam[:,1]/cam[:,2]*K[1,1]+K[1,2]).astype(int)
    h,wd=rgb.shape[:2]
    m=(u>=1)&(u<wd-1)&(v>=1)&(v<h-1)
    for uu,vv in zip(u[m],v[m]):
        rgb[vv-1:vv+2,uu-1:uu+2]=colour
    return rgb, int(m.sum())
pw,info=perceived_world(env); ow,_=export(env.sim)
a,na=overlay(pw,[255,60,60]); b,nb=overlay(ow,[60,160,255])
raw=np.asarray(env._last_obs[f"{CAM_NAMES[AGENTVIEW]}_image"])[::-1]
frame=np.concatenate([raw,a,b],axis=1)
os.makedirs(REPO+"/videos",exist_ok=True)
imageio.imwrite(REPO+"/videos/world_depth_vs_oracle.png", frame)
imageio.mimsave(REPO+"/videos/world_depth_vs_oracle.mp4",[frame]*40,fps=20,macro_block_size=1)
print(f"perceived {info['points']} pts -> {info['voxels_total']} voxels; "
      f"projected: perceived {na}, oracle {nb}")
print("wrote videos/world_depth_vs_oracle.png and .mp4  (raw | perceived RED | oracle BLUE)")
